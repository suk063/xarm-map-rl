import numpy as np
import robosuite.utils.transform_utils as T


def format_ndarray(array, digits=3):
    return "[" + ', '.join(f"{array[i]: .{digits}f}" for i in range(len(array))) + "]"


def relative_quat_method_1(robot0_eef_pos, robot0_eef_quat, cube_pos, cube_quat):
    world_pose_in_gripper = T.pose_inv(T.pose2mat((robot0_eef_pos, robot0_eef_quat)))
    cube_pose = T.pose2mat((cube_pos, cube_quat))
    rel_pose = T.pose_in_A_to_pose_in_B(cube_pose, world_pose_in_gripper)
    rel_pos, rel_quat = T.mat2pose(rel_pose)
    return rel_quat


def relative_quat_method_2(robot0_eef_quat, cube_quat):
    return T.quat_distance(robot0_eef_quat, cube_quat)


def compare_relative_quat_methods():
    n_samples = 1000

    # Generate random position and quaternion for robot0 end-effector and cube
    robot0_eef_pos = np.random.uniform(-0.5, 0.5, size=(n_samples, 3))
    robot0_eef_quat = np.random.uniform(-1, 1, size=(n_samples, 4))
    robot0_eef_quat /= np.linalg.norm(robot0_eef_quat, axis=1, keepdims=True)

    cube_pos = np.random.uniform(-0.5, 0.5, size=(n_samples, 3))
    cube_quat = np.random.uniform(-1, 1, size=(n_samples, 4))
    cube_quat /= np.linalg.norm(cube_quat, axis=1, keepdims=True)

    # Compare the two methods for calculating the relative quaternion
    mismatch_count = 0
    for i in range(n_samples):
        # print(f"Sample {i}: norm robot0_eef_quat: {np.linalg.norm(robot0_eef_quat[i])}, norm cube_quat: {np.linalg.norm(cube_quat[i])}")
        rel_quat_1 = relative_quat_method_1(robot0_eef_pos[i], robot0_eef_quat[i], cube_pos[i], cube_quat[i])
        rel_quat_2 = relative_quat_method_2(robot0_eef_quat[i], cube_quat[i])
        
        if not np.allclose(rel_quat_1, rel_quat_2):
            mismatch_count += 1
            # print(f"Sample {i} | Quaternions not equal: {rel_quat_1} vs {rel_quat_2}")
            # break
    print(f"Sample {n_samples} | Quaternions not equal: {rel_quat_1} vs {rel_quat_2}")
    print(f"Total mismatch count: {mismatch_count} out of {n_samples} samples")


if __name__ == "__main__":
    compare_relative_quat_methods()