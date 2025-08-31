import argparse
import os
import numpy as np
import cv2
import transforms3d as t3d
import plotly.graph_objects as go

class CustomIntrinsics:
    """A simple class to hold camera intrinsic values."""
    def __init__(self, width, height, fx, fy, ppx, ppy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy

def calculate_intrinsics_from_fov(width, height, fov_x_deg, fov_y_deg):
    """Calculates camera intrinsics based on FOV and image dimensions."""
    fov_x_rad = np.deg2rad(fov_x_deg)
    fov_y_rad = np.deg2rad(fov_y_deg)
    fx = width / (2 * np.tan(fov_x_rad / 2))
    fy = height / (2 * np.tan(fov_y_rad / 2))
    ppx = width / 2
    ppy = height / 2
    return CustomIntrinsics(width, height, fx, fy, ppx, ppy)

def deproject_to_pointcloud(depth_img, intrinsics, depth_scale=1000.0):
    """Converts a depth image to a point cloud using camera intrinsics."""
    height, width = depth_img.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Depth values are in mm, so we scale by 1000 to get meters
    z = depth_img.astype(float) / depth_scale
    x = (u - intrinsics.ppx) * z / intrinsics.fx
    y = (v - intrinsics.ppy) * z / intrinsics.fy
    
    points = np.dstack((x, y, z)).reshape(-1, 3)
    return points

def create_frame_trace(matrix, size=0.05):
    """Creates Plotly traces for a 3D coordinate frame."""
    origin = matrix[:3, 3]
    x_axis = origin + matrix[:3, 0] * size
    y_axis = origin + matrix[:3, 1] * size
    z_axis = origin + matrix[:3, 2] * size

    traces = []
    traces.append(go.Scatter3d(x=[origin[0], x_axis[0]], y=[origin[1], x_axis[1]], z=[origin[2], x_axis[2]], mode='lines', line=dict(color='red', width=4), showlegend=False))
    traces.append(go.Scatter3d(x=[origin[0], y_axis[0]], y=[origin[1], y_axis[1]], z=[origin[2], y_axis[2]], mode='lines', line=dict(color='green', width=4), showlegend=False))
    traces.append(go.Scatter3d(x=[origin[0], z_axis[0]], y=[origin[1], z_axis[1]], z=[origin[2], z_axis[2]], mode='lines', line=dict(color='blue', width=4), showlegend=False))
    return traces

def main(dataset_path):
    """Loads a dataset and visualizes the combined point cloud and end-effector poses using Plotly."""
    
    # Define paths
    rgb_dir = os.path.join(dataset_path, "rgb")
    depth_dir = os.path.join(dataset_path, "depth")
    pose_dir = os.path.join(dataset_path, "pose")

    if not os.path.isdir(dataset_path):
        print(f"Error: Dataset directory not found at '{dataset_path}'")
        return

    # Define crop parameters and calculate intrinsics based on them
    CROP_W, CROP_H = 480, 480
    FOV_X, FOV_Y = 57, 57
    intrinsics = calculate_intrinsics_from_fov(CROP_W, CROP_H, FOV_X, FOV_Y)
    
    # Placeholder for arm-to-camera transformation, assuming identity
    arm2cam = np.eye(4)

    # Get file lists and sort them
    rgb_files = sorted([os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith('.png')])
    depth_files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')])
    pose_files = sorted([os.path.join(pose_dir, f) for f in os.listdir(pose_dir) if f.endswith('.npy')])

    all_points = []
    all_colors = []
    plotly_traces = []

    frame = 0

    if len(rgb_files) > 0:
        # --- Process only the first frame ---
        color_img = cv2.imread(rgb_files[frame], cv2.IMREAD_COLOR)
        depth_img = cv2.imread(depth_files[frame], cv2.IMREAD_UNCHANGED)

        if color_img is not None and depth_img is not None:
            # --- Image Preprocessing ---
            if color_img.shape[:2] != depth_img.shape[:2]:
                depth_img = cv2.resize(depth_img, (color_img.shape[1], color_img.shape[0]), interpolation=cv2.INTER_NEAREST)

            orig_h, orig_w = color_img.shape[:2]
            x_offset = (orig_w - CROP_W) // 2
            y_offset = (orig_h - CROP_H) // 2
            
            if x_offset >= 0 and y_offset >= 0:
                color_img_cropped = color_img[y_offset:y_offset+CROP_H, x_offset:x_offset+CROP_W]
                depth_img_cropped = depth_img[y_offset:y_offset+CROP_H, x_offset:x_offset+CROP_W]

                points_cam = deproject_to_pointcloud(depth_img_cropped, intrinsics)
                colors = cv2.cvtColor(color_img_cropped, cv2.COLOR_BGR2RGB).reshape(-1, 3)

                valid_indices = (points_cam[:, 2] > 0.1) & (points_cam[:, 2] < 5.0)
                points_cam = points_cam[valid_indices]
                colors = colors[valid_indices]
                
                # Point cloud is already in camera frame, no transformation needed.
                
                # Filter points to be within the [-1m, 1m] cube around the camera origin
                in_box_indices = np.all((points_cam >= -1.0) & (points_cam <= 1.0), axis=1)
                points_cam = points_cam[in_box_indices]
                colors = colors[in_box_indices]

                all_points.append(points_cam)
                all_colors.append(colors)

                pose = np.load(pose_files[frame])
                translation = pose[:3]
                rotation = t3d.euler.euler2mat(pose[3], pose[4], pose[5], 'sxyz')
                eef_pose_matrix_arm = np.eye(4)
                eef_pose_matrix_arm[:3, :3] = rotation
                eef_pose_matrix_arm[:3, 3] = translation
                
                # Transform EEF pose from arm frame to camera frame for consistent visualization
                eef_pose_matrix_cam = arm2cam @ eef_pose_matrix_arm
                plotly_traces.extend(create_frame_trace(eef_pose_matrix_cam))
        # --- End of first frame processing ---

    if not all_points:
        print("Error: No valid data in the first frame. Nothing to visualize.")
        return

    combined_points = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)
    
    num_points = combined_points.shape[0]
    max_points = 50000 
    if num_points > max_points:
        print(f"Downsampling from {num_points} to {max_points} points for visualization.")
        indices = np.random.choice(num_points, max_points, replace=False)
        combined_points = combined_points[indices]
        combined_colors = combined_colors[indices]

    pcd_trace = go.Scatter3d(
        x=combined_points[:, 0],
        y=combined_points[:, 1],
        z=combined_points[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=combined_colors / 255.0),
        name='Point Cloud'
    )
    plotly_traces.append(pcd_trace)
    
    plotly_traces.extend(create_frame_trace(np.eye(4), size=0.1))

    fig = go.Figure(data=plotly_traces)
    fig.update_layout(
        title=f"First Frame Point Cloud Visualization: {os.path.basename(dataset_path)}",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube',
            xaxis=dict(range=[-1, 1],),
            yaxis=dict(range=[-1, 1],),
            zaxis=dict(range=[-1,  1],)
        ),
        showlegend=True
    )
    
    print("Visualizing the first frame in Plotly.")
    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a multi-viewpd point cloud from a saved dataset using Plotly.")
    parser.add_argument("dataset_path", type=str, help="Path to the root directory of the dataset.")
    args = parser.parse_args()
    main(args.dataset_path)
