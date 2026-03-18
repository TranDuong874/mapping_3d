import os, torch, numpy as np, cv2, open3d as o3d, gc, resource
from glob import glob
import pyorbslam3
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.image import load_images

# --- GROUNDED MAPPING SETTINGS ---
VOXEL_SIZE = 0.02
MIN_CONF = 1.5
SPATIAL_STEP = 1
# ------------------------------

def solve_umeyama(P, Q):
    """
    Solves for s, R, t such that s*R*P + t approx Q
    P: (N, 3) Source (MASt3R local)
    Q: (N, 3) Target (SLAM world)
    """
    if P.shape[0] < 3: return 1.0, np.eye(3), np.zeros(3)
    
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    
    P_c = P - mu_p
    Q_c = Q - mu_q
    
    var_p = np.mean(np.sum(P_c**2, axis=1))
    
    H = P_c.T @ Q_c / P.shape[0]
    U, S, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    s = np.trace(np.diag(S)) / var_p if var_p > 1e-6 else 1.0
    t = mu_q - s * R @ mu_p
    
    return s, R, t

def get_ram_usage():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

def main():
    vocab = 'dependency/ORB_SLAM3/Vocabulary/ORBvoc.txt'
    settings = 'dependency/ORB_SLAM3/Examples/Monocular/TUM-VI.yaml'
    model_pth = 'models/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    image_dir = 'dataset/dataset-corridor1_512_16/mav0/cam0/data'
    output_path = 'outputs/tum_dense_grounded.ply'
    os.makedirs('outputs', exist_ok=True)
    device = 'cuda'

    # Fisheye Rectification
    K_orig = np.array([[190.978, 0, 254.932], [0, 190.973, 256.897], [0, 0, 1]])
    D = np.array([0.00348239, 0.000715035, -0.00205324, 0.000202937])
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K_orig, D, (512,512), np.eye(3), balance=0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_orig, D, np.eye(3), new_K, (512,512), cv2.CV_16SC2)

    print("Loading MASt3R...")
    model = AsymmetricMASt3R.from_pretrained(model_pth).to(device).eval()
    
    print("Initializing ORB-SLAM3 (Headless)...")
    slam = pyorbslam3.System(vocab, settings, pyorbslam3.Sensor.MONOCULAR, False)
    
    global_pcd = o3d.geometry.PointCloud()
    img_files = sorted(glob(os.path.join(image_dir, '*.png')))
    last_kf_data = None
    processed_kfs = 0

    for i, path in enumerate(img_files):
        img_raw = cv2.imread(path)
        if img_raw is None: continue
        
        timestamp = float(os.path.basename(path)[:-4]) / 1e9
        res = slam.track_monocular(img_raw, timestamp)
        
        pose_valid = res.get('pose_valid', False)
        is_kf = res.get('is_keyframe', False)
        state = res.get('tracking_state')

        if pose_valid and state == 2 and is_kf:
            obs = slam.get_tracked_observations()
            if last_kf_data is None:
                last_kf_data = {'img_path': path, 'uvs': obs['keypoints_uv'], 'pts_world': obs['world_points_xyz']}
                continue

            print(f"Update @ frame {i} (KF {processed_kfs}) | Grounding to {len(obs['world_points_xyz'])} pts")
            
            # Prep Images
            im1 = cv2.remap(cv2.imread(last_kf_data['img_path']), map1, map2, cv2.INTER_LINEAR)
            im2 = cv2.remap(img_raw, map1, map2, cv2.INTER_LINEAR)
            
            # Temporary files for dust3r loader
            cv2.imwrite('t1.png', im1); cv2.imwrite('t2.png', im2)
            
            inputs = load_images(['t1.png', 't2.png'], size=320)
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):
                    v1 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs[0].items()}
                    v2 = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs[1].items()}
                    v1['true_shape'] = torch.from_numpy(np.array(v1['true_shape'])).to(device)
                    v2['true_shape'] = torch.from_numpy(np.array(v2['true_shape'])).to(device)
                    
                    r1, _ = model(v1, v2)
                    pts3d = r1['pts3d'].squeeze(0)
                    
                    # Align MASt3R local space to SLAM world space
                    ref_uvs, ref_world = last_kf_data['uvs'], last_kf_data['pts_world']
                    if len(ref_world) >= 3:
                        u_norm = (ref_uvs[:, 0] * (320/512)).astype(int).clip(0, 319)
                        v_norm = (ref_uvs[:, 1] * (240/512)).astype(int).clip(0, 239)
                        pts_mast3r = pts3d[v_norm, u_norm].cpu().numpy()
                        
                        scale, R, t = solve_umeyama(pts_mast3r, ref_world)
                        
                        conf = r1['conf'].squeeze(0).view(-1)
                        dense_pts = pts3d.view(-1, 3)
                        rgb = (v1['img'].squeeze(0).permute(1,2,0)*0.5+0.5).view(-1, 3)
                        
                        mask = (conf > MIN_CONF)
                        dense_pts = dense_pts[mask][::SPATIAL_STEP].cpu().numpy()
                        rgb = rgb[mask][::SPATIAL_STEP].cpu().numpy()
                        
                        dense_pts_world = (scale * (R @ dense_pts.T)).T + t
                        
                        local_pcd = o3d.geometry.PointCloud()
                        local_pcd.points = o3d.utility.Vector3dVector(dense_pts_world)
                        local_pcd.colors = o3d.utility.Vector3dVector(rgb)
                        local_pcd, _ = local_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                        
                        global_pcd.points.extend(local_pcd.points)
                        global_pcd.colors.extend(local_pcd.colors)
                        global_pcd = global_pcd.voxel_down_sample(VOXEL_SIZE)
                
            last_kf_data = {'img_path': path, 'uvs': obs['keypoints_uv'], 'pts_world': obs['world_points_xyz']}
            processed_kfs += 1
            torch.cuda.empty_cache(); gc.collect()

            if processed_kfs % 10 == 0:
                o3d.io.write_point_cloud(output_path, global_pcd)
                print(f"Checkpoint saved: {len(global_pcd.points)} points")

        if get_ram_usage() > 13.0:
            print("RAM limit reached. Stopping.")
            break

    if len(global_pcd.points) > 0:
        global_pcd, _ = global_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
        o3d.io.write_point_cloud(output_path, global_pcd)
        print(f"Mapping Finished: {len(global_pcd.points)} points saved to {output_path}")
    
    slam.shutdown()
    
    # Cleanup temp files
    for f in ['t1.png', 't2.png', 'tmp1.png', 'tmp2.png']:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__": main()
