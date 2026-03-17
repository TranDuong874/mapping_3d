#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "MapPoint.h"
#include "System.h"

namespace py = pybind11;

namespace {

struct ImuMeasurement {
    double timestamp_s;
    std::array<float, 3> linear_acceleration_m_s2;
    std::array<float, 3> angular_velocity_rad_s;
};

bool is_finite_se3(const Sophus::SE3f& pose) {
    const auto translation = pose.translation();
    const auto quaternion = pose.unit_quaternion();
    return std::isfinite(translation.x()) && std::isfinite(translation.y()) &&
           std::isfinite(translation.z()) && std::isfinite(quaternion.w()) &&
           std::isfinite(quaternion.x()) && std::isfinite(quaternion.y()) &&
           std::isfinite(quaternion.z());
}

std::string tracking_state_name(int tracking_state) {
    switch (tracking_state) {
        case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
            return "SYSTEM_NOT_READY";
        case ORB_SLAM3::Tracking::NO_IMAGES_YET:
            return "NO_IMAGES_YET";
        case ORB_SLAM3::Tracking::NOT_INITIALIZED:
            return "NOT_INITIALIZED";
        case ORB_SLAM3::Tracking::OK:
            return "OK";
        case ORB_SLAM3::Tracking::RECENTLY_LOST:
            return "RECENTLY_LOST";
        case ORB_SLAM3::Tracking::LOST:
            return "LOST";
        case ORB_SLAM3::Tracking::OK_KLT:
            return "OK_KLT";
        default:
            return "UNKNOWN";
    }
}

cv::Mat image_from_numpy(const py::array& image) {
    py::buffer_info info = image.request();
    if (info.ndim == 2) {
        cv::Mat mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_8UC1,
            info.ptr
        );
        return mat.clone();
    }
    if (info.ndim == 3 && info.shape[2] == 3) {
        cv::Mat mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_8UC3,
            info.ptr
        );
        return mat.clone();
    }
    throw std::invalid_argument("image must have shape (H, W) or (H, W, 3) with dtype uint8");
}

cv::Mat depth_from_numpy(const py::array& depth) {
    py::buffer_info info = depth.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("depth image must have shape (H, W)");
    }

    cv::Mat result;
    if (py::isinstance<py::array_t<float>>(depth)) {
        cv::Mat mat(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_32F,
            info.ptr
        );
        result = mat.clone();
    } else if (py::isinstance<py::array_t<double>>(depth)) {
        cv::Mat mat64(
            static_cast<int>(info.shape[0]),
            static_cast<int>(info.shape[1]),
            CV_64F,
            info.ptr
        );
        mat64.convertTo(result, CV_32F);
    } else {
        throw std::invalid_argument("depth image must have dtype float32 or float64");
    }
    return result;
}

std::vector<ORB_SLAM3::IMU::Point> convert_imu(const std::vector<ImuMeasurement>& imu_measurements) {
    std::vector<ORB_SLAM3::IMU::Point> converted;
    converted.reserve(imu_measurements.size());
    for (const ImuMeasurement& sample : imu_measurements) {
        converted.emplace_back(
            cv::Point3f(
                sample.linear_acceleration_m_s2[0],
                sample.linear_acceleration_m_s2[1],
                sample.linear_acceleration_m_s2[2]
            ),
            cv::Point3f(
                sample.angular_velocity_rad_s[0],
                sample.angular_velocity_rad_s[1],
                sample.angular_velocity_rad_s[2]
            ),
            sample.timestamp_s
        );
    }
    std::sort(
        converted.begin(),
        converted.end(),
        [](const ORB_SLAM3::IMU::Point& lhs, const ORB_SLAM3::IMU::Point& rhs) {
            return lhs.t < rhs.t;
        }
    );
    return converted;
}

py::object pose_matrix_to_numpy(const Sophus::SE3f& pose) {
    if (!is_finite_se3(pose)) {
        return py::none();
    }

    const Eigen::Matrix4f matrix = pose.matrix();
    py::array_t<float> result({4, 4});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < 4; ++row) {
        for (py::ssize_t col = 0; col < 4; ++col) {
            output(row, col) = matrix(row, col);
        }
    }
    return result;
}

py::tuple pose_translation_to_tuple(const Sophus::SE3f& pose) {
    const auto translation = pose.translation();
    return py::make_tuple(translation.x(), translation.y(), translation.z());
}

py::tuple pose_quaternion_to_tuple(const Sophus::SE3f& pose) {
    const auto quaternion = pose.unit_quaternion();
    return py::make_tuple(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
}

py::dict make_tracking_result(
    const Sophus::SE3f& pose,
    int tracking_state,
    bool is_keyframe
) {
    py::dict result;
    result["tracking_state"] = tracking_state;
    result["tracking_state_name"] = tracking_state_name(tracking_state);
    result["is_keyframe"] = is_keyframe;
    result["pose_valid"] = is_finite_se3(pose);
    result["pose_matrix"] = pose_matrix_to_numpy(pose);
    if (is_finite_se3(pose)) {
        result["translation_xyz"] = pose_translation_to_tuple(pose);
        result["quaternion_wxyz"] = pose_quaternion_to_tuple(pose);
    } else {
        result["translation_xyz"] = py::none();
        result["quaternion_wxyz"] = py::none();
    }
    return result;
}

py::array_t<float> vector3_list_to_numpy(const std::vector<Eigen::Vector3f>& points) {
    py::array_t<float> result({static_cast<py::ssize_t>(points.size()), py::ssize_t(3)});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < output.shape(0); ++index) {
        output(index, 0) = points[index].x();
        output(index, 1) = points[index].y();
        output(index, 2) = points[index].z();
    }
    return result;
}

py::array_t<float> keypoints_to_numpy(const std::vector<cv::KeyPoint>& keypoints) {
    py::array_t<float> result({static_cast<py::ssize_t>(keypoints.size()), py::ssize_t(2)});
    auto output = result.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < output.shape(0); ++index) {
        output(index, 0) = keypoints[index].pt.x;
        output(index, 1) = keypoints[index].pt.y;
    }
    return result;
}

py::dict tracked_observations_to_python(
    const std::vector<cv::KeyPoint>& keypoints,
    const std::vector<ORB_SLAM3::MapPoint*>& map_points
) {
    std::vector<cv::KeyPoint> filtered_keypoints;
    std::vector<Eigen::Vector3f> filtered_world_points;
    const std::size_t num_entries = std::min(keypoints.size(), map_points.size());
    filtered_keypoints.reserve(num_entries);
    filtered_world_points.reserve(num_entries);

    for (std::size_t index = 0; index < num_entries; ++index) {
        ORB_SLAM3::MapPoint* map_point = map_points[index];
        if (!map_point || map_point->isBad()) {
            continue;
        }

        const Eigen::Vector3f world_point = map_point->GetWorldPos();
        if (!std::isfinite(world_point.x()) || !std::isfinite(world_point.y()) ||
            !std::isfinite(world_point.z())) {
            continue;
        }

        filtered_keypoints.push_back(keypoints[index]);
        filtered_world_points.push_back(world_point);
    }

    py::dict result;
    result["keypoints_uv"] = keypoints_to_numpy(filtered_keypoints);
    result["world_points_xyz"] = vector3_list_to_numpy(filtered_world_points);
    return result;
}

class OrbSlam3System {
public:
    OrbSlam3System(
        const std::string& vocabulary_path,
        const std::string& settings_path,
        ORB_SLAM3::System::eSensor sensor,
        bool use_viewer = false,
        int init_frame = 0,
        const std::string& sequence = std::string()
    )
        : system_(vocabulary_path, settings_path, sensor, use_viewer, init_frame, sequence),
          shut_down_(false) {}

    ~OrbSlam3System() {
        shutdown();
    }

    py::dict track_monocular(
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& image,
        double timestamp_s,
        const std::vector<ImuMeasurement>& imu_measurements = {},
        const std::string& filename = std::string()
    ) {
        cv::Mat image_mat = image_from_numpy(image);
        std::vector<ORB_SLAM3::IMU::Point> imu_points = convert_imu(imu_measurements);
        Sophus::SE3f pose;
        int tracking_state = -1;
        bool is_keyframe = false;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            pose = system_.TrackMonocular(image_mat, timestamp_s, imu_points, filename);
            tracking_state = system_.GetTrackingState();
            is_keyframe = system_.WasLastFrameKeyFrame();
        }
        return make_tracking_result(pose, tracking_state, is_keyframe);
    }

    py::dict track_stereo(
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& left_image,
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& right_image,
        double timestamp_s,
        const std::vector<ImuMeasurement>& imu_measurements = {},
        const std::string& filename = std::string()
    ) {
        cv::Mat left_mat = image_from_numpy(left_image);
        cv::Mat right_mat = image_from_numpy(right_image);
        std::vector<ORB_SLAM3::IMU::Point> imu_points = convert_imu(imu_measurements);
        Sophus::SE3f pose;
        int tracking_state = -1;
        bool is_keyframe = false;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            pose = system_.TrackStereo(left_mat, right_mat, timestamp_s, imu_points, filename);
            tracking_state = system_.GetTrackingState();
            is_keyframe = system_.WasLastFrameKeyFrame();
        }
        return make_tracking_result(pose, tracking_state, is_keyframe);
    }

    py::dict track_rgbd(
        const py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>& image,
        const py::array& depth_image,
        double timestamp_s,
        const std::vector<ImuMeasurement>& imu_measurements = {},
        const std::string& filename = std::string()
    ) {
        cv::Mat image_mat = image_from_numpy(image);
        cv::Mat depth_mat = depth_from_numpy(depth_image);
        std::vector<ORB_SLAM3::IMU::Point> imu_points = convert_imu(imu_measurements);
        Sophus::SE3f pose;
        int tracking_state = -1;
        bool is_keyframe = false;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            pose = system_.TrackRGBD(image_mat, depth_mat, timestamp_s, imu_points, filename);
            tracking_state = system_.GetTrackingState();
            is_keyframe = system_.WasLastFrameKeyFrame();
        }
        return make_tracking_result(pose, tracking_state, is_keyframe);
    }

    int get_tracking_state() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_running();
        return system_.GetTrackingState();
    }

    std::string get_tracking_state_name() {
        return tracking_state_name(get_tracking_state());
    }

    py::array_t<float> get_current_map_points() {
        std::vector<Eigen::Vector3f> points;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            points = system_.GetCurrentMapPointPositions();
        }
        return vector3_list_to_numpy(points);
    }

    py::array_t<float> get_tracked_keypoints() {
        std::vector<cv::KeyPoint> keypoints;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            keypoints = system_.GetTrackedKeyPointsUn();
        }
        return keypoints_to_numpy(keypoints);
    }

    py::dict get_tracked_observations() {
        std::vector<cv::KeyPoint> keypoints;
        std::vector<ORB_SLAM3::MapPoint*> map_points;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ensure_running();
            keypoints = system_.GetTrackedKeyPointsUn();
            map_points = system_.GetTrackedMapPoints();
        }
        return tracked_observations_to_python(keypoints, map_points);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_running();
        system_.Reset();
    }

    void reset_active_map() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_running();
        system_.ResetActiveMap();
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shut_down_) {
            return;
        }
        system_.Shutdown();
        shut_down_ = true;
    }

    bool is_shutdown() const {
        return shut_down_;
    }

    float get_image_scale() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensure_running();
        return system_.GetImageScale();
    }

private:
    void ensure_running() const {
        if (shut_down_) {
            throw std::runtime_error("ORB-SLAM3 system has been shut down");
        }
    }

    ORB_SLAM3::System system_;
    mutable std::mutex mutex_;
    bool shut_down_;
};

}  // namespace

PYBIND11_MODULE(_orbslam3, module) {
    module.doc() = "Direct pybind11 bindings for ORB-SLAM3";

    py::enum_<ORB_SLAM3::System::eSensor>(module, "Sensor")
        .value("MONOCULAR", ORB_SLAM3::System::MONOCULAR)
        .value("STEREO", ORB_SLAM3::System::STEREO)
        .value("RGBD", ORB_SLAM3::System::RGBD)
        .value("IMU_MONOCULAR", ORB_SLAM3::System::IMU_MONOCULAR)
        .value("IMU_STEREO", ORB_SLAM3::System::IMU_STEREO)
        .value("IMU_RGBD", ORB_SLAM3::System::IMU_RGBD)
        .export_values();

    py::class_<ImuMeasurement>(module, "ImuMeasurement")
        .def(
            py::init<
                double,
                const std::array<float, 3>&,
                const std::array<float, 3>&
            >(),
            py::arg("timestamp_s"),
            py::arg("linear_acceleration_m_s2"),
            py::arg("angular_velocity_rad_s")
        )
        .def_readwrite("timestamp_s", &ImuMeasurement::timestamp_s)
        .def_readwrite("linear_acceleration_m_s2", &ImuMeasurement::linear_acceleration_m_s2)
        .def_readwrite("angular_velocity_rad_s", &ImuMeasurement::angular_velocity_rad_s);

    py::class_<OrbSlam3System>(module, "System")
        .def(
            py::init<
                const std::string&,
                const std::string&,
                ORB_SLAM3::System::eSensor,
                bool,
                int,
                const std::string&
            >(),
            py::arg("vocabulary_path"),
            py::arg("settings_path"),
            py::arg("sensor"),
            py::arg("use_viewer") = false,
            py::arg("init_frame") = 0,
            py::arg("sequence") = std::string()
        )
        .def("track_monocular", &OrbSlam3System::track_monocular, py::arg("image"), py::arg("timestamp_s"), py::arg("imu_measurements") = std::vector<ImuMeasurement>{}, py::arg("filename") = std::string())
        .def("track_stereo", &OrbSlam3System::track_stereo, py::arg("left_image"), py::arg("right_image"), py::arg("timestamp_s"), py::arg("imu_measurements") = std::vector<ImuMeasurement>{}, py::arg("filename") = std::string())
        .def("track_rgbd", &OrbSlam3System::track_rgbd, py::arg("image"), py::arg("depth_image"), py::arg("timestamp_s"), py::arg("imu_measurements") = std::vector<ImuMeasurement>{}, py::arg("filename") = std::string())
        .def("get_tracking_state", &OrbSlam3System::get_tracking_state)
        .def("get_tracking_state_name", &OrbSlam3System::get_tracking_state_name)
        .def("get_current_map_points", &OrbSlam3System::get_current_map_points)
        .def("get_tracked_keypoints", &OrbSlam3System::get_tracked_keypoints)
        .def("get_tracked_observations", &OrbSlam3System::get_tracked_observations)
        .def("reset", &OrbSlam3System::reset)
        .def("reset_active_map", &OrbSlam3System::reset_active_map)
        .def("shutdown", &OrbSlam3System::shutdown)
        .def("is_shutdown", &OrbSlam3System::is_shutdown)
        .def("get_image_scale", &OrbSlam3System::get_image_scale);
}
