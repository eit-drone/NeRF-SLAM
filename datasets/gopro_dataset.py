import numpy as np
import cv2
import os

from datasets.dataset import *
from datasets.relay.mqtt_recv import MQTTVideoStream


class GoProDataset(Dataset):
    def __init__(self, args, device, resize_images=True):
        super().__init__("GoPro", args, device)
        self.timestamp = 0
        self.capture = MQTTVideoStream()
        self.capture.listen_for_frames()

        self.calib = self._get_cam_calib()
        self.resize_images = resize_images
        self.viz = False

        if self.resize_images:
            self.output_image_size = [360, 480]  # h, w
            h0, w0 = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = self.output_image_size[0] * self.output_image_size[1]
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)
        else:
            self.w1 = self.calib.resolution.width
            self.h1 = self.calib.resolution.height

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
            
            if self.viz:
                cv2.imshow("Img Raw", frame)
                cv2.waitKey(1)

            image = cv2.resize(frame, (self.w1, self.h1))

            if self.viz:
                cv2.imshow("Img Resize", image)
                cv2.waitKey(1)

            #image = cv2.undistort(
            #    image,
            #    self.calib.camera_model.matrix(),
            #    self.calib.distortion_model.get_distortion_as_vector(),
            #)

            # if self.viz:
            #     cv2.imshow("Img Undistort", image)
            #     cv2.waitKey(1)

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

        print("Got image", self.timestamp, "from GoPro")
        return {
            "k": np.arange(self.timestamp - 1, self.timestamp),
            "t_cams": np.array(timestamps),
            "poses": np.array(poses),
            "depths": np.array(depths),
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
            439.20174027063297,
            439.20174027063297 / 1.0002917258561743,
            476.8955661316992,
            266.1759474051889,
        )

        # k1, k2, p1, p2, p3 = (
        #     -6.201146315205147e-05,
        #     0.0002860389617367069,
        #     -0.27947200915285003,
        #     0.1275640762597187,
        #     -0.03410419310123914,
        # )
        k1, k2, p1, p2, p3 = 0, 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = 5  # 25 FPS but skipping every 5 frames

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
        self.capture.shutdown()
