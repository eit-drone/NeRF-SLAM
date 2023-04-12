
import numpy as np
import cv2
import os

from datasets.dataset import *

class WebcamDataset(Dataset):

    def __init__(self, args, device, resize_images=True):   
        super().__init__("Webcam", args, device)
        self.capture = cv2.VideoCapture(0)

        self.calib = self._get_cam_calib()
        self.resize_images = resize_images

        if self.resize_images:
            self.output_image_size = [315, 420] # h, w 
            h0, w0  = self.calib.resolution.height, self.calib.resolution.width
            total_output_pixels = (self.output_image_size[0] * self.output_image_size[1])
            self.h1 = int(h0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.w1 = int(w0 * np.sqrt(total_output_pixels / (h0 * w0)))
            self.h1 = self.h1 - self.h1 % 8
            self.w1 = self.w1 - self.w1 % 8
            self.calib.camera_model.scale_intrinsics(self.w1 / w0, self.h1 / h0)
            self.calib.resolution = Resolution(self.w1, self.h1)

    def stream(self):
        timestamps = []
        poses      = []
        images     = []
        depths     = []
        calibs     = []

        got_image = False
        while not got_image:
            # Wait for a coherent pair of frames: depth and color
            try:
                frames = self.pipeline.wait_for_frames()
            except Exception as e: 
                print(e)
                continue

            color_frame = frames.get_color_frame()

            if not color_frame:
                print("No color frame parsed.")
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            if self.resize_images:
                color_image = cv2.resize(color_image, (self.w1, self.h1))

            self.timestamp += 1
            if self.args.img_stride > 1 and self.timestamp % self.args.img_stride == 0:
                # Software imposed fps to rate_hz/img_stride
                continue

            timestamps += [self.timestamp]
            poses      += [np.eye(4)] # We don't have poses
            images     += [color_image]
            calibs     += [self.calib]
            got_image  = True

        return {"k":      np.arange(self.timestamp-1,self.timestamp),
                "t_cams": np.array(timestamps),
                "poses":  np.array(poses),
                "images": np.array(images),
                "calibs": np.array(calibs),
                "is_last_frame": False, #TODO
                }
    
    def _get_cam_calib(self):
        """ intrinsics: 
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

        w, h = intrinsics.width, intrinsics.height
        fx, fy, cx, cy= intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy

        distortion_coeffs = intrinsics.coeffs
        distortion_model  = intrinsics.model
        k1, k2, p1, p2 = 0, 0, 0, 0
        body_T_cam0 = np.eye(4)
        rate_hz = self.rate_hz

        resolution  = Resolution(w, h)
        pinhole0    = PinholeCameraModel(fx, fy, cx, cy)
        distortion0 = RadTanDistortionModel(k1, k2, p1, p2)

        aabb        = (2*np.array([[-2, -2, -2], [2, 2, 2]])).tolist() # Computed automatically in to_nerf()
        depth_scale = 1.0 # TODO # Since we multiply as gt_depth *= depth_scale, we need to invert camera["scale"]

        return CameraCalibration(body_T_cam0, pinhole0, distortion0, rate_hz, resolution, aabb, depth_scale)

    def shutdown(self):
        # Stop streaming
        self.capture.release()