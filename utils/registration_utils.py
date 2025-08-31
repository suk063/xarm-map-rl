import open3d as o3d
import numpy as np
import copy
import time
import transforms3d as t3d


def draw_registration_result(source, target, transformation, window_name="Open3D"):
    source_temp = o3d.geometry.PointCloud(source.points)
    target_temp = o3d.geometry.PointCloud(target.points)
    source_temp.paint_uniform_color([1, 0.2, 0])
    target_temp.paint_uniform_color([0, 0.2, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)


def preprocess_point_cloud(pcd, voxel_size, print_info=True):
    if print_info:
        print("  :: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    if print_info:
        print("  :: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    if print_info:
        print("  :: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size, source_path=None, target_path=None, partial_source=False, print_info=True):
    print(":: Load two point clouds and disturb initial pose.")

    if source_path is None or target_path is None:
        # l is the measured size of the physical green cube (in meters)
        l = 0.055
        resolution = l/100
        xyz = np.mgrid[-l/2:l/2+resolution:resolution, -l/2:l/2+resolution:resolution, -l/2:l/2+resolution:resolution].transpose(1,2,3,0).reshape(-1,3)

    if target_path is not None:
        target = o3d.io.read_point_cloud(target_path)
    else:
        target_xyz = xyz[np.max(np.abs(xyz), axis=1) == l/2]
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_xyz)

    if source_path is not None:
        source = o3d.io.read_point_cloud(source_path)
    else:
        if partial_source:
            source_xyz = xyz[np.max(xyz, axis=1) == l/2]
        else:
            source_xyz = xyz[np.max(np.abs(xyz), axis=1) == l/2]
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_xyz)
        trans = np.eye(4)
        trans[:3,:3] = t3d.euler.euler2mat(0, 0, np.pi/6)
        source.transform(trans)

    draw_registration_result(source, target, np.identity(4), window_name="Initial Pointclouds")

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, print_info=print_info)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, print_info=print_info)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size, print_info=True):
    distance_threshold = voxel_size
    if print_info:
        print("\n:: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

    
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size, print_info=True):
    distance_threshold = voxel_size * 0.5
    if print_info:
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def test_global_registration(source_path=None, target_path=None, partial_source=False, print_info=True):
    # Using opencv kinfu's resolution from cv::kinfu::Params::defaultParams()
    # https://github.com/opencv/opencv_contrib/blob/442085f85994ccbb276d070c263176b15afd93ff/modules/rgbd/src/kinfu.cpp
    # Value in meters
    voxel_size = 3/512
    voxel_size /= 4
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_path=source_path, target_path=target_path, partial_source=partial_source, print_info=print_info)

    start = time.time()
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size,
                                                print_info=print_info
                                                )
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print(result_ransac, "\n\n")
    draw_registration_result(source_down, target_down, result_ransac.transformation, window_name="RANSAC")


    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size,
                                                   print_info=print_info
                                                   )
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    draw_registration_result(source_down, target_down, result_fast.transformation, window_name="Fast Global Registration")


def execute_icp_point2point(source, target, source_down, target_down, 
                            source_fpfh, target_fpfh, voxel_size, threshold, print_info=True,
                            trans_init=None):
    if trans_init is None:
        trans_init = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size,
                                                    print_info=print_info
                                                    ).transformation
    if print_info:
        print("Apply point-to-point ICP")
    result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result
    

def execute_icp_point2plane(source, target, source_down, target_down, 
                            source_fpfh, target_fpfh, voxel_size, threshold, print_info=True,
                            trans_init=None):
    if trans_init is None:
        trans_init = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size,
                                                    print_info=print_info
                                                    ).transformation
    if print_info:
        print("Apply point-to-plane ICP")
    result = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def test_registration(source_path=None, target_path=None, partial_source=False, print_info=True):
    voxel_size = 3/512
    voxel_size /= 4
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(voxel_size, source_path=source_path, target_path=target_path, partial_source=partial_source, print_info=print_info)

    start = time.time()
    trans_init = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size,
                                                    print_info=print_info
                                                    ).transformation
    threshold = 0.01
    if print_info:
        print("Initial global registration using RANSAC took %.3f sec.\n" % (time.time() - start))
        evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
        print(evaluation, "\n")
    draw_registration_result(source_down, target_down, trans_init, window_name="Initial alignment with RANSAC")
    
    start = time.time()
    result_p2p = execute_icp_point2point(source, target, source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size, threshold,
                                        print_info=print_info,
                                        trans_init=trans_init
                                        )
    if print_info:
        print("point2point ICP took %.3f sec.\n" % (time.time() - start))
    print(result_p2p, "\n\n")
    draw_registration_result(source, target, result_p2p.transformation, window_name="ICP Point2Point")

    start = time.time()
    result_p2l = execute_icp_point2plane(source, target, source_down, target_down,
                                        source_fpfh, target_fpfh,
                                        voxel_size, threshold,
                                        print_info=print_info,
                                        trans_init=trans_init
                                        )
    if print_info:
        print("point2plane ICP took %.3f sec.\n" % (time.time() - start))
    print(result_p2l, "\n\n")
    draw_registration_result(source, target, result_p2l.transformation, window_name="ICP Point2Plane")
    

def get_relative_tranformation(source, target, voxel_size = 3/512, algo="ICP", euler_angles=False, print_info=False):
    '''
        For given source and target pointclouds, this returns the transformation to apply to source 
        in order to match the target.
        Input:
            source: Source pointcloud
            target: Target point cloud
            voxel_size: Resolution used to downsample pointclouds
            algo: 'RANSAC' or 'FGR' (Fast Global Registration)
            euler_angles: Return euler angles 
            print_info: print parameter information used for registration 
        Returns:
            If euler_angles is False, return relative transformation SE(3) matrix
            If euler_angles is True, return euler angles of relative transformation around x, y, z axes (calculated as 'sxyz')
        
        Tips for visualization - 
            View original source and target:
                draw_registration_result(source, target, np.eye(4))
            View transformed source and target:
                draw_registration_result(source, target, get_relative_tranformation(source, target))
    '''
    assert algo in ["FGR", "RANSAC", "ICP", "ICP2"], "algo needs to be 'FGR', 'RANSAC', 'ICP', or 'ICP2'"
    
    # Default voxel size is from opencv kinfu - cv::kinfu::Params::defaultParams()
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size, print_info=print_info)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size, print_info=print_info)
    result = None
    if algo == "FGR":
        result = execute_fast_global_registration(source_down, target_down,
                                                  source_fpfh, target_fpfh,
                                                  voxel_size, print_info=print_info)
    elif algo == "RANSAC":
        result = execute_global_registration(source_down, target_down,
                                             source_fpfh, target_fpfh,
                                             voxel_size, print_info=print_info)
    elif algo == "ICP":
        result = execute_icp_point2point(source, target, source_down, target_down,
                                             source_fpfh, target_fpfh,
                                             voxel_size, 0.01, print_info=print_info)
    elif algo == "ICP2":
        result = execute_icp_point2plane(source, target, source_down, target_down,
                                             source_fpfh, target_fpfh,
                                             voxel_size, 0.01, print_info=print_info)
    # print(f"\nresult fitness: {result.fitness}\n")
    if euler_angles:
        return t3d.euler.mat2euler(result.transformation[:3, :3]), result.fitness
    else:
        return result.transformation, result.fitness


def pairwise_colored_registration_for_multiway(source, target,
                                               max_correspondence_distance_coarse, max_correspondence_distance_fine):
    voxel_radius = [max_correspondence_distance_coarse, 
                    (max_correspondence_distance_coarse+max_correspondence_distance_fine)/2, 
                    max_correspondence_distance_fine]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    # print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter, radius, scale])

        # print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        # print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        # print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                      relative_rmse=1e-6,
                                                                      max_iteration=iter))
        current_transformation = result_icp.transformation
        # print(result_icp)
    transformation_icp = current_transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        transformation_icp)
    return transformation_icp, information_icp


def pairwise_registration_for_multiway(source, target, 
                                       max_correspondence_distance_coarse, max_correspondence_distance_fine):
    # print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def multiway_registration(pcds, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine, use_color=False):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)

    if use_color:
        pairwise_registration_algo = pairwise_colored_registration_for_multiway
    else:
        pairwise_registration_algo = pairwise_registration_for_multiway
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration_algo(
                pcds[source_id], pcds[target_id],
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
            # print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def load_point_clouds(pcd_paths, voxel_size=3/512):
    pcds = []
    for pcd_path in pcd_paths:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)
    return pcds


def get_multiway_registered_pcd(pcd_paths, visualize=False, use_color=False):
    voxel_size = 3/512/4
    pcds_down = load_point_clouds(pcd_paths, voxel_size=voxel_size)
    if visualize:
        o3d.visualization.draw_geometries(pcds_down, window_name="Initial Pointclouds")

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = multiway_registration(pcds_down,
                                           max_correspondence_distance_coarse,
                                           max_correspondence_distance_fine, use_color=use_color)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds_down)):
        # print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds_down[point_id]
    if visualize:
        o3d.visualization.draw_geometries([pcd_combined], window_name="Registered Pointclouds")
    return pcd_combined


def handle_cube_symmetry(euler_angles, in_degrees=True):
    euler_angles_deg = np.array(euler_angles)
    if not in_degrees:
        euler_angles_deg = np.array(euler_angles) * 180/np.pi
    euler_angles_deg = euler_angles_deg % 90
    euler_angles_deg[euler_angles_deg > 45] = 90 - euler_angles_deg[euler_angles_deg > 45]
    return euler_angles_deg


def perform_registration(object_pcd=None, model_pcd=None, fast_mode=False, handle_cube_symmetries=False):
    '''
        Returns euler angles (in degrees) of relative transformation between object_pcd and model_pcd
        Ranges - roll: [-180, 180], pitch: [-90, 90], yaw: [-180, 180]
    '''
    # TODO: Change hardcoded model file paths
    if object_pcd is None:
        object_pcd = o3d.io.read_point_cloud("./object_realtime_color.ply")
    rel_transformation, fitness_score = get_relative_tranformation(object_pcd, model_pcd, print_info=False)
    if not fast_mode:
        # draw_registration_result(object_pcd, model_pcd, np.eye(4), window_name="Initial Pointclouds")
        draw_registration_result(object_pcd, model_pcd, rel_transformation, window_name="Registered Pointclouds")
    euler_angles_deg = np.array(t3d.euler.mat2euler(rel_transformation[:3,:3])) * 180/np.pi
    if handle_cube_symmetries:
        euler_angles_deg = handle_cube_symmetry(euler_angles_deg)
    if not fast_mode:
        print(f"Relative transformation euler angles are: {euler_angles_deg} degrees, \ntranslation: {rel_transformation[:3,3]}, \nfitness score: {fitness_score}")
    
    euler_angles_translations_deg = np.concatenate([euler_angles_deg, rel_transformation[:3,3]])
    return euler_angles_translations_deg, fitness_score


def standardize_quaternion(quat):
    '''
        Standardize an xyzw quaternion to have a positive scalar component
    '''
    if quat[3] < 0:
        quat = -quat
    return quat


if __name__ == "__main__":
    # source_path = target_path = None
    # source_path = "/home/erl-tianyu/xArm_camera/kinfu_scans/just_cuboid_y30.ply"
    # target_path = "/home/erl-tianyu/xArm_camera/kinfu_scans/just_cuboid.ply"

    real_scene_path = "/home/erl-tianyu/xArm_robosuite/cube_realtime_color.ply"
    real_model_path = "/home/erl-tianyu/xArm_robosuite/cube_model_manual_scanalign.ply"
    # real_model_path = "/home/erl-tianyu/xArm_robosuite/cube_realtime_color_trimmed_1.ply"

    # Fake model (target), Fake scene (source)
    # test_registration(source_path=None, target_path=None, partial_source=True)

    # # Fake model (target), Real scene (source)
    # test_registration(source_path=real_scene_path, target_path=None)
    
    # # Real model (target), Fake scene (source)
    # test_registration(source_path=None, target_path=real_model_path, partial_source=True)

    # Real model (target), Real scene (source)
    test_registration(source_path=real_scene_path, target_path=real_model_path)
    
