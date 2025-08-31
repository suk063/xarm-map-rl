import os
import multiprocessing as mp
import time

import cv2
import pyrealsense2 as rs
import numpy as np
import cv2.aruco as aruco

from utils.debug_utils import format_ndarray
from ruamel.yaml import YAML

# Define red color range in HSV
LOWER_RED_1 = np.array([0,160,60], dtype=np.uint8)
UPPER_RED_1 = np.array([5,255,255], dtype=np.uint8)       
LOWER_RED_2 = np.array([174,100,60], dtype=np.uint8)
UPPER_RED_2 = np.array([179,255,255], dtype=np.uint8)       

# Define green color range in HSV
# LOWER_GREEN = np.array([30, 60, 50], dtype=np.uint8)
LOWER_GREEN = np.array([30, 80, 55], dtype=np.uint8)
UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_dir = os.path.join(parent_dir, "configs")

yaml = YAML(typ='safe')
config_params = yaml.load(open(config_dir + "/config_gui.yaml", "r"))

task = config_params["task"]
cube_offset = np.array(config_params["camera_cube_offset"][task])
video_dir = os.path.join(parent_dir, "scripts/outputs/video_logs", time.strftime("%Y-%m-%d"), task)


class Camera:
    """
    Interface for using Intel RealSense D435 
    Currently we can use camera to detect a green cube
    or detect Aruco markers for camera extrinsics calibration
    """
    def __init__(self, debug=False, save_video=False, imsize=(640, 480), xarm=None):

        self.debug = debug
        self.save_video = save_video
        self.xarm = xarm

        # For saving posed RGBD dataset
        self.save_count = 0
        self.dataset_dir = None
        self.should_quit = False

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        profile = self.pipeline.start(config)        
        
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        # colorizer = rs.colorizer()
        self.filters = [
            rs.disparity_transform(),
            rs.spatial_filter(),
            rs.temporal_filter(),
            rs.disparity_transform(False)
        ]

        # Align depth image to color image
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        time.sleep(2)
        
        # Extrinsics to be loaded
        self.cam2arm = None
        self.arm2cam = None

        self.imsize = imsize
        video_imsize = (imsize[0]*2, imsize[1])  # Concatenated color and depth images
        if save_video:
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.writer = cv2.VideoWriter(
                video_dir + f"/{time.strftime('%H-%M-%S')}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                15, 
                video_imsize
            )

        # For debugging with simultaneous detection of green and red objects
        if self.debug:
            self.prev_color = None
            self.prev_rect_pos = None
            self.prev_uvd = None
            self.prev_xyz = None

        print(f"Using camera cube offset {cube_offset} for task {task}")


    @staticmethod
    def get_bbox(frame, color='g'):
        cnt = Camera.get_contour(frame, color=color)
        if cnt is not None:
            return cv2.boundingRect(cnt)
        else:
            return None

    @staticmethod
    def get_contour(frame, color='g'):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if color == 'g':
            mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        elif color == 'r':
            mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
            mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
            mask = mask1 + mask2
        # res = cv2.bitwise_and(frame, frame, mask=mask)    

        # Perform adaptive thresholding to handle varying lighting conditions
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, adaptive_threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(adaptive_threshold, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours of the largest green/red object
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key = cv2.contourArea)
            if cv2.contourArea(cnt) > 350:
                return cnt
        return None

    def fetch_image(self, middle_crop_imsize=None, border_crops=(0,0,0,0)):
        success, frames = self.pipeline.try_wait_for_frames()
        if not success:
            return None

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Depth frame post-processing
        depth_frame = self.decimate.process(depth_frame)
        for f in self.filters:
            depth_frame = f.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()  

        # Grab new intrinsics (may be changed by decimation)
        depth_intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrin.width, depth_intrin.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not middle_crop_imsize is None:
            assert middle_crop_imsize[0] <= color_image.shape[0] and middle_crop_imsize[1] <= color_image.shape[1], \
                "Crop image size is larger than the color image size"
            assert middle_crop_imsize[0]//2 <= depth_image.shape[0] and middle_crop_imsize[1]//2 <= depth_image.shape[1], \
                "Crop image size is larger than the depth image size (accounting for 2x resolution difference)"


            color_top = (color_image.shape[0] - middle_crop_imsize[0]) // 2
            color_left = (color_image.shape[1] - middle_crop_imsize[1]) // 2
            color_image = color_image[color_top:color_top + middle_crop_imsize[0], 
                                            color_left:color_left + middle_crop_imsize[1]]


            depth_crop_h, depth_crop_w = middle_crop_imsize[0] // 2, middle_crop_imsize[1] // 2
            depth_top = (depth_image.shape[0] - depth_crop_h) // 2
            depth_left = (depth_image.shape[1] - depth_crop_w) // 2
            depth_image = depth_image[depth_top:depth_top + depth_crop_h,
                                            depth_left:depth_left + depth_crop_w]
        
        if border_crops != (0,0,0,0):
            top, right, bottom, left = border_crops
            H, W = color_image.shape[:2]

            assert top >= 0 and bottom >= 0 and left >= 0 and right >= 0, \
                "Border crop values must be non-negative"
            assert top + bottom < H and left + right < W, \
                "Border crop boundaries exceed image dimensions"

            color_image = color_image[top:H-bottom, left:W-right]
            depth_image = depth_image[top//2:H//2-bottom//2, left//2:W//2-right//2]

        if self.imsize != (640, 480):
            assert self.imsize[0] <= color_image.shape[0] and self.imsize[1] <= color_image.shape[1], \
                "Resize image size is larger than the cropped color image size"
            assert self.imsize[0]//2 <= depth_image.shape[0] and self.imsize[1]//2 <= depth_image.shape[1], \
                "Resize image size is larger than the cropped depth image size (accounting for 2x resolution difference)"

            color_image = cv2.resize(color_image, self.imsize, interpolation=cv2.INTER_LINEAR)
            depth_image = cv2.resize(depth_image, self.imsize, interpolation=cv2.INTER_LINEAR)

        if self.debug:
            # Concatenate color and depth images
            depth_colormap = cv2.cvtColor(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLOR_GRAY2BGR)
            depth_colormap = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
            rgb_depth_concat = np.hstack((color_image, depth_colormap))
            cv2.imshow("RGBD Image", rgb_depth_concat)

            if self.save_video:
                self.writer.write(rgb_depth_concat)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if self.dataset_dir is None:
                    self.dataset_dir = os.path.join(parent_dir, "scripts/outputs/posed_rgbd_dataset", time.strftime("%Y-%m-%d-%H-%M-%S"))
                    os.makedirs(os.path.join(self.dataset_dir, "rgb"), exist_ok=True)
                    os.makedirs(os.path.join(self.dataset_dir, "depth"), exist_ok=True)
                    os.makedirs(os.path.join(self.dataset_dir, "pose"), exist_ok=True)
                
                filename = f"{self.save_count:06d}"
                rgb_file = os.path.join(self.dataset_dir, "rgb", f"{filename}.png")
                depth_file = os.path.join(self.dataset_dir, "depth", f"{filename}.png")
                pose_file = os.path.join(self.dataset_dir, "pose", f"{filename}.npy")
                
                cv2.imwrite(rgb_file, color_image)
                cv2.imwrite(depth_file, depth_image)
                
                if self.xarm:
                    code, pose = self.xarm.arm.get_position(is_radian=True)
                    if code == 0:
                        np.save(pose_file, pose)
                        print(f"Saved frame {self.save_count} to {self.dataset_dir}")
                        self.save_count += 1
                    else:
                        print(f"Failed to get EEF pose, error code {code}")
                else:
                    print("XArm object not provided. Cannot save pose.")

            elif key == ord('q'):
                self.should_quit = True

        return color_image, depth_image

    def detect(self, arm_frame=False, color='g'):
        """
        Detects green/red object 3D position in camera frame
        """
        
        success, frames = self.pipeline.try_wait_for_frames()
        if not success:
            return None

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Depth frame post-processing
        depth_frame = self.decimate.process(depth_frame)
        for f in self.filters:
            depth_frame = f.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()  

        # Grab new intrinsics (may be changed by decimation)
        depth_intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrin.width, depth_intrin.height

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        bbox = Camera.get_bbox(color_image, color=color)
        if bbox is None:
            return None

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        s1, s2 = color_image.shape[0] / h, color_image.shape[1] / w 
        u, v = int((bbox[0] + bbox[2] / 2)), int((bbox[1] + bbox[3] / 2))
        d = depth_frame.get_distance(int(u / s1), int(v / s2)) 
        
        # pixel to 3D point in camera frame
        color_intrin = rs.video_stream_profile(color_frame.profile).get_intrinsics()
        x, y, z = rs.rs2_deproject_pixel_to_point(color_intrin, [u, v], d)

        # normalize z
        # z = np.sqrt(d**2 - x**2 - y**2)   

        if self.debug:
            cv_params_g = {"rect_color" : (0, 255, 0),
                        "text1_org" : (100, 110),
                        "text2_org" : (100, 130),
                        "text_color" : (50, 170, 50)}
            cv_params_r = {"rect_color" : (0, 0, 255),
                        "text1_org" : (100, 60),
                        "text2_org" : (100, 80),
                        "text_color" : (50, 50, 170)}
            if color == 'g':
                cv_params = cv_params_g
                other_params = cv_params_r
            elif color == 'r':
                cv_params = cv_params_r
                other_params = cv_params_g
            cv2.rectangle(color_image, p1, p2, cv_params["rect_color"], 2, 1)
            cv2.putText(
                color_image, 
                f"Pixel {u, v}, depth {d*100:.1f} cm", 
                cv_params["text1_org"], cv2.FONT_HERSHEY_SIMPLEX, 0.75, cv_params["text_color"], 2
            )
            cv2.putText(
                color_image, 
                f"Point {x*100:.1f}, {y*100:.1f}, {z*100:.1f}", 
                cv_params["text2_org"], cv2.FONT_HERSHEY_SIMPLEX, 0.75, cv_params["text_color"], 2
            )

            if self.prev_color is not None and self.prev_color != color:
                cv2.rectangle(color_image, *self.prev_rect_pos, other_params["rect_color"], 2, 1)
                cv2.putText(
                    color_image, 
                    f"Pixel {self.prev_uvd[0], self.prev_uvd[1]}, depth {self.prev_uvd[2]*100:.1f} cm", 
                    other_params["text1_org"], cv2.FONT_HERSHEY_SIMPLEX, 0.75, other_params["text_color"], 2
                )
                cv2.putText(
                    color_image, 
                    f"Point {self.prev_xyz[0]*100:.1f}, {self.prev_xyz[1]*100:.1f}, {self.prev_xyz[2]*100:.1f}", 
                    other_params["text2_org"], cv2.FONT_HERSHEY_SIMPLEX, 0.75, other_params["text_color"], 2
                )

            self.prev_color = color
            self.prev_rect_pos = (p1, p2)
            self.prev_uvd = (u, v, d)
            self.prev_xyz = (x, y, z)

            cv2.imshow("Object detection", color_image)
            if self.save_video:
                self.writer.write(color_image)
            cv2.waitKey(1)

        pos = np.array([x, y, z])
        if arm_frame:
            if self.cam2arm is None:
                self.cam2arm, self.arm2cam = load_extrinsics()
            pos = self.cam2arm[:3, :3] @ pos + self.cam2arm[:3, -1] + cube_offset
        return pos

    def close(self):
        try:
            if self.save_video:
                self.writer.release()
            self.pipeline.stop()
        except RuntimeError:
            pass


    def detect_aruco(self):
        """
        Detects the aruco marker in camera frame
        """

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_intrinsics = rs.video_stream_profile(color_frame.profile).get_intrinsics()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        parameters =  aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, markerIds, rejectedCandidates = detector.detectMarkers(gray)

        if len(corners)!=0:
            print(f"Found markers: {markerIds}, {corners}")
            if self.debug:
                color_image = aruco.drawDetectedMarkers(color_image, corners)
                # cv2.putText(color_image, f"Center: {x:.3f} {y:.3f} {z:.3f}", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                # cv2.putText(color_image, f"Point: {u, v}", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                cv2.imshow("Aruco Detection", color_image) 
                cv2.waitKey(1)           

            u, v = np.average(corners[0][0], axis=0)
            u, v = int(u), int(v)
            d = depth_frame.get_distance(u, v)

            center = [0,0,0]
            if d != 0:
                x, y ,z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [u, v], d)
                z = np.sqrt(d**2 - x**2 - y**2)
                center = [x, y, z]

            return center

        else:
            return None

    def flush(self):
        for _ in range(50):
            frames = self.pipeline.wait_for_frames()
    
    def __del__(self):
        self.close()


def save_extrinsics(cam2arm, arm2cam, filename='extrinsics.npz'):
    np.savez(filename, cam2arm=cam2arm, arm2cam=arm2cam)

def load_extrinsics(filename='extrinsics.npz'):
    try:
        extrinsics = np.load(filename)
    except FileNotFoundError as e:
        print("ERROR: Extrinsics not found. Please run camera_extrinsics_calibration.py first to save extrinsics (required to convert 3D locations to arm frame).\n")
        exit(1)
    cam2arm = extrinsics['cam2arm']
    arm2cam = extrinsics['arm2cam']
    return cam2arm, arm2cam
    

def camera_thread(buffer_position, stop_event_camera):
    """
    This function should be run as an mp process, always on and detect green object
    buffer_position: mp.Queue to store detected object position
    stop_event_camera: mp.Event that can be set to stop the process
    """

    print("Camera pid:", os.getpid())
    
    # writer = cv2.VideoWriter(
    #     'outpy.avi',
    #     cv2.VideoWriter_fourcc('M','J','P','G'), 
    #     60, 
    #     (640, 480)
    # )

    # Set up depth postprocessing filters
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 1)
    filters = [
        rs.disparity_transform(),
        rs.spatial_filter(),
        rs.temporal_filter(),
        rs.disparity_transform(False)
    ]

    pipeline = rs.pipeline()
    config = rs.config()

    # Set high frame rate to sync color and depth
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    align_to = rs.stream.color
    align = rs.align(align_to)   
    time.sleep(2)

    # Start streaming
    pipeline.start(config)

    while not stop_event_camera.is_set():
    # for i in range(600):
        frames = pipeline.wait_for_frames()

        # Perform depth frame post-processing before alignment
        frames = decimate.process(frames)
        for f in filters:
            frames = f.process(frames)
        frames = rs.composite_frame(frames)
    
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Grab new intrinsics (may be changed by decimation)
        depth_intrin = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrin.width, depth_intrin.height  

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        bbox = Camera.get_bbox(color_image)

        if bbox is None:
            if buffer_position.full():
                out = buffer_position.get(block=False)
            buffer_position.put(None)

        else:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))       

            s1, s2 = color_image.shape[0] / h, color_image.shape[1] / w 
            u, v = int((bbox[0] + bbox[2] / 2)), int((bbox[1] + bbox[3] / 2))
            d = depth_frame.get_distance(int(u / s1), int(v / s2))      

            # pixel to 3D point in camera frame
            color_intrin = rs.video_stream_profile(color_frame.profile).get_intrinsics()
            x, y, z = rs.rs2_deproject_pixel_to_point(color_intrin, [u, v], d)  

            if buffer_position.full():
                out = buffer_position.get(block=False)
                # print(f"{out[0]:.3f}, {out[1]:.3f}, {out[2]:.3f}")
            buffer_position.put([x, y, z])    

            cv2.rectangle(color_image, p1, p2, (255,0,0), 2, 1)
            cv2.putText(
                color_image, 
                f"Pixel {u, v}, depth {d*100:.2f} cm", 
                (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2
            )
            cv2.putText(
                color_image, 
                f"Point {x*100:.2f}, {y*100:.2f}, {z*100:.2f}", 
                (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2
            )
            cv2.imshow("Object detection", color_image) 
            cv2.waitKey(1)


    plt.plot(obj_g_pos_history)
    if detect_red_parallely:
        plt.figure(figsize=(18, 3))
        plt.plot(obj_r_pos_history)
    if calc_pose:
        plt.figure(figsize=(18, 3))
        plt.plot(obj_quat_history)
    plt.show()

    print(f"\nAverage green cube position: {np.mean(obj_g_pos_history, axis=0)}")
    print(f"Standard deviation of cube position: {np.std(obj_g_pos_history, axis=0)}\n")


if __name__ == '__main__':
    from devices.xarm6 import XArmControl
    xarm = XArmControl(ip="192.168.1.242", simulated=False)
    cam = Camera(debug=True, save_video=True, xarm=xarm)
    while True:
        cam.fetch_image()
        if cam.should_quit:
            break
    cam.close()
    xarm.close()