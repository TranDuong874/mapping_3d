import torch
from depth_anything_3.api import DepthAnything3
from time import perf_counter

# Load model from Hugging Face Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DepthAnything3.from_pretrained("depth-anything/da3-small")
model = model.to(device=device)

# Run inference on images
import os

image_roots = '../dataset/dataset-corridor1_512_16/mav0/cam1/data'

images = [os.path.join(image_roots, file) for file in os.listdir(image_roots)][:300] 

start = perf_counter()
prediction = model.inference(
    images,
    export_dir="output",
    export_format="glb"  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
)
detla = perf_counter() - start

print(f"Execution time: {detla}")
print(f"Time per frame: {detla / len(images)}")

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32
