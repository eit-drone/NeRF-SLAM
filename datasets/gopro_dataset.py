import numpy as np
import cv2
import os

from datasets.dataset import *
from datasets.relay.mqtt_recv import MQTTVideoStream


class GoProDataset(Dataset):
    def __init__(self, args, device, resize_images=True):
        super().__init__("GoPro", args, device)
        self.capture = MQTTVideoStream()

        self.original_calib = self._get_cam_calib()
        self.calib = self._get_cam_calib()
        self.resize_images = resize_images

        if self.resize_images:
            self.output_image_size = [315, 420]  # h, w
            h0, w0 = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = self.output_image_size[0] * self.output_image_size[1]
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

    def stream(self):
        timestamps = []
        poses = []
        images = []
        depths = []
        calibs = []

        got_image = False
        while not got_image:
            frame = self.capture.get_frame()

            if frame is None:
                continue

            # Undistort img
            image = cv2.undistort(
                frame,
                self.original_calib.camera_model.matrix(),
                self.original_calib.distortion_model.get_distortion_as_vector(),
            )

            if self.resize_images:
                image = cv2.resize(image, (self.w1, self.h1))

            self.timestamp += 1
            if self.args.img_stride > 1 and self.timestamp % self.args.img_stride == 0:
                # Software imposed fps to rate_hz/img_stride
                continue

            timestamps += [self.timestamp]
            poses += [np.eye(4)]  # We don't have poses
            images += [image]
            depths += [None]
            calibs += [self.calib]
            got_image = True

        return {
            "k": np.arange(self.timestamp - 1, self.timestamp),
            "t_cams": np.array(timestamps),
            "poses": np.array(poses),
            "images": np.array(images),
            "calibs": np.array(calibs),
            "is_last_frame": False,  # TODO
        }

    def _get_cam_calib(self):
        """intrinsics:
        model	Distortion model of the image
        coeffs	Distortion coefficients
        fx	    Focal length of the image plane, as a multiple of pixel width
        fy	    Focal length of the image plane, as a multiple of pixel height
        ppx	    Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
        ppy	    Vertical coordinate of the principal point of the image, as a pixel offset from the top edge
        height	Height of the image in pixels
        width	Width of the image in pixels
        """
        # Calibration tool: https://github.com/urbste/OpenImuCameraCalibrator
        # TODO: Get calibration from camera with PINHOLE_RADIAL_TANGENTIAL model

        w, h = 960, 540
        fx, fy, cx, cy = (
            450.3699397559044,
            450.3699397559044,
            480.0374873071809,
            263.598993630644,
        )

        k1, k2, p1, p2, p3 = (
            0.0010069560892659898,
            -0.00015277026676884705,
            -0.29818479449943347,
            0.13188878385230152,
            -0.0086043436353433,
        )
        body_T_cam0 = np.eye(4)
        rate_hz = 5 # 25 FPS but skipping every 5 frames

        resolution = Resolution(w, h)
        pinhole0 = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2, p3)

        aabb = (
            2 * np.array([[-2, -2, -2], [2, 2, 2]])
        ).tolist()  # Computed automatically in to_nerf()
        depth_scale = 1.0  # Not used

        return CameraCalibration(
            body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale
        )

    def shutdown(self):
        # Stop streaming
        self.capture.release()
