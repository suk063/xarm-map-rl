"""
Calibrate camera to xArm extrinsics.
After calibration, extrinsics are stored in extrinsics.npz 
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from devices.xarm6 import XArmControl
from devices.camera import Camera, save_extrinsics


def main():

    xarm_control = XArmControl(ip="192.168.1.242", tcp_z_offset=50)
    camera = Camera(debug=True)

    centers = []
    # Calibration points in xArm robot base frame - generated using `scripts/generate_calibration_points.py`
    # This is the xArm end-effector position (center of last joint, not the gripper!)
    arm_coords = np.array([
    [250,   -150,   90  ], 
    [250,   -150,   180 ], 
    [250,   -150,   270 ], 
    [250,      0,   270 ], 
    [250,      0,   180 ], 
    [250,      0,   90  ], 
    [250,    150,   90  ], 
    [250,    150,   180 ], 
    [250,    150,   270 ], 
    [400,    150,   270 ], 
    [400,    150,   180 ], 
    [400,    150,   90  ], 
    [400,      0,   90  ], 
    [400,      0,   180 ], 
    [400,      0,   270 ], 
    [400,   -150,   270 ], 
    [400,   -150,   180 ], 
    [400,   -150,   90  ], 
    [550,   -150,   90  ], 
    [550,   -150,   180 ], 
    [550,   -150,   270 ], 
    [550,      0,   270 ], 
    [550,      0,   180 ], 
    [550,      0,   90  ], 
    [550,    150,   90  ], 
    [550,    150,   180 ], 
    [550,    150,   270 ]
    ]) / 1000 + np.array([0, 0, 0.1])
    N = len(arm_coords)

    # Marker is on front side of the gripper, adjust offsets in x and z directions
    marker_coords = (arm_coords + np.array([0.022, 0, -0.055])) 

    # Move end-effector to each point and detect marker in camera frame
    for pt in arm_coords:
        x, y, z = pt
        xarm_control.set_eef_position(x=x, y=y, z=z)
        
        aruco_location = None
        while aruco_location is None:
            aruco_location = camera.detect_aruco()
        centers.append(aruco_location)
    centers = np.array(centers)

    # point (in camera coord) = arm2cam * point (in arm coord)
    marker_coords_c = np.concatenate([marker_coords.T, np.ones((1, N))], axis=0)
    arm2cam = centers.T @ np.linalg.pinv(marker_coords_c) 

    # point (in arm coord) = cam2arm * point (in camera coord)
    centers_c = np.concatenate([centers.T, np.ones((1, N))], axis=0)
    cam2arm = marker_coords.T @ np.linalg.pinv(centers_c) 

    print("Camera to arm transform:\n", cam2arm)
    print("Arm to camera transform:\n", arm2cam)
    print("Sanity Test:")   
    for marker_pt, image_pt in zip(marker_coords, centers):
        print("-------------------")
        print("Expected:", marker_pt)
        print("Result:", cam2arm[:3, :3] @ image_pt + cam2arm[:3, -1])
        print("Expected:", image_pt)
        print("Result:", arm2cam[:3, :3] @ marker_pt + arm2cam[:3, -1])  
    
    cam_frame_errors_mean = np.mean(np.abs((arm2cam[:3, :3] @ marker_coords.T).T + arm2cam[:3, -1] - centers), axis=0)
    arm_frame_errors_mean = np.mean(np.abs((cam2arm[:3, :3] @ centers.T).T + cam2arm[:3, -1] - marker_coords), axis=0)
    print("\nMean error in camera frame:", cam_frame_errors_mean)
    print("Mean error in arm frame:", arm_frame_errors_mean)

    save_extrinsics(cam2arm, arm2cam)

    xarm_control.close()


if __name__ == '__main__':
    main()
