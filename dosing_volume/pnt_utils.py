import open3d as o3d
import wrs.vision.depth_camera.pcd_data_adapter as pda
import numpy as np
from shapely.geometry.polygon import Polygon
import wrs.basis.robot_math as rm
from wrs.basis.trimesh import Trimesh
from itertools import chain
import wrs.vision.depth_camera.util_functions as dcuf
from file_sys import load_pickle, workdir_vision, load_json
import copy
import wrs.modeling.geometric_model as gm

# RACK_HARD_TEMPLATE = load_pickle(workdir_vision / "template" / "rack_hard_temp")
# RACK_SOFT_TEMPLATE = load_pickle(workdir_vision / "template" / "rack_soft_temp")


def segment_table_plane(pcd_np, distance_threshold=0.03, ransac_n=10):
    """Segment a plane from point cloud data. Look at:
    http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Plane-segmentation
    for details
    """
    pct_o3d = pda.nparray_to_o3dpcd(pcd_np)
    plane_model, inliers = pct_o3d.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=1000)
    inlier_cloud = pct_o3d.select_by_index(inliers)
    outlier_cloud = pct_o3d.select_by_index(inliers, invert=True)
    return pda.o3dpcd_to_nparray(inlier_cloud), pda.o3dpcd_to_nparray(outlier_cloud)


def extract_pcd_by_yolo(pcd,
                        img_shape,
                        bound_lt, bound_rb,
                        enlarge_detection_ratio=.2):
    """Extract the rack according to detection of the yolo"""
    img_height, img_width = img_shape[0], img_shape[1]
    pcd_reshape = pcd.reshape(img_height, img_width, 3)
    enlarge_detection = (np.array(bound_rb) - np.array(bound_lt)) * enlarge_detection_ratio / 2
    bound_lt = (np.array(bound_lt) - enlarge_detection)
    bound_lt = np.clip(bound_lt, 0, np.inf).astype(np.int)
    bound_rb = (np.array(bound_rb) + enlarge_detection)
    bound_rb = np.clip(bound_rb, 0, np.inf).astype(np.int)
    rack_pcd = pcd_reshape[
               bound_lt[1]:bound_rb[1],
               bound_lt[0]:bound_rb[0]
               ].reshape(-1, 3)
    return rack_pcd


def extract_pcd_by_range(pcd, x_range=None, y_range=None, z_range=None, origin_pos=np.zeros(3), origin_rot=np.eye(3),
                         toggle_debug=False):
    origin_frame = rm.homomat_from_posrot(origin_pos, origin_rot)
    pcd_align = rm.homomat_transform_points(np.linalg.inv(origin_frame), pcd)
    pcd_ind = np.ones(len(pcd_align), dtype=bool)
    if x_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 0] >= x_range[0]) & (pcd_align[:, 0] <= x_range[1])
    if y_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 1] >= y_range[0]) & (pcd_align[:, 1] <= y_range[1])
    if z_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 2] >= z_range[0]) & (pcd_align[:, 2] <= z_range[1])
    if toggle_debug:
        from basis.trimesh.primitives import Box
        ext_pcd = pcd_align[np.where(pcd_ind)[0]]
        x_range = x_range if x_range is not None else [ext_pcd[:, 0].min(), ext_pcd[:, 0].max()]
        y_range = y_range if y_range is not None else [ext_pcd[:, 1].min(), ext_pcd[:, 1].max()]
        z_range = z_range if z_range is not None else [ext_pcd[:, 2].min(), ext_pcd[:, 2].max()]
        extract_region = Box(
            box_extents=[(x_range[1] - x_range[0]), (y_range[1] - y_range[0]), (z_range[1] - z_range[0]), ],
            box_center=[(x_range[1] + x_range[0]) / 2, (y_range[1] + y_range[0]) / 2, (z_range[1] + z_range[0]) / 2, ])
        bx_gm = gm.GeometricModel(extract_region)
        bx_gm.set_rgba([1, 0, 0, .3])
        bx_gm.set_homomat(origin_frame)
        bx_gm.attach_to(base)
    return np.where(pcd_ind)[0]


def detect_keypoint_iss_from_model(mesh_path):
    mesh_path = str(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(9000)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    return np.asarray(keypoints.points)


def __preprocess_point_cloud(pcd, voxel_size, normal_radius=2, feature_radius=5):
    vvvv = load_json(workdir_vision / "template" / "radius")
    normal_radius = vvvv["normal_radius"]
    feature_radius = vvvv["feature_radius"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # step 1: estimate normals
    down_radius_normal = voxel_size * normal_radius
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
    # step 2: extract FPFH descriptor
    radius_feature = voxel_size * feature_radius
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=int(len(pcd_down.points) / 11)))
    return pcd_down, pcd_fpfh


def __preprocess_point_cloud_iss(pcd, voxel_size):
    pcd_down = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    # step 1: estimate normals
    down_radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=down_radius_normal, max_nn=30))
    # step 2: extract FPFH descriptor
    radius_feature = voxel_size * 4
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def __draw_registration_result(source_o3d, target_o3d, transform=None, corresponding_set=None):
    source_temp = copy.deepcopy(source_o3d)
    target_temp = copy.deepcopy(target_o3d)

    source_temp_np = np.asarray(source_temp.points)
    target_temp_np = np.asarray(target_temp.points)
    source_temp_normal = np.asarray(source_temp.normals)
    target_temp_normal = np.asarray(target_temp.normals)

    base.clear_render2()

    for nn in range(0, len(source_temp_normal), 5):
        gm.gen_arrow(source_temp_np[nn], source_temp_np[nn] + source_temp_normal[nn] * 0.01,
                     thickness=0.0009,
                     rgba=[0, 1, 0, 1])._objpdnp.reparentTo(base.render2)

    for nn in range(0, len(target_temp_normal), 5):
        gm.gen_arrow(target_temp_np[nn], target_temp_np[nn] + target_temp_normal[nn] * 0.01,
                     thickness=0.0009,
                     rgba=[0, 0, 1, 1])._objpdnp.reparentTo(base.render2)

    pcd_source_origin = gm.gen_pointcloud(
        source_temp_np,
        [[1, 0.706, 0, 1]])
    pcd_source_origin._objpdnp.reparentTo(base.render2)
    if transform is not None:
        pcd_source_trans = gm.GeometricModel(pcd_source_origin.copy())
        pcd_source_trans.set_homomat(transform)
        pcd_source_trans._objpdnp.reparentTo(base.render2)
    gm.gen_pointcloud(
        target_temp_np,
        [[0, 0.651, 0.929, 1]])._objpdnp.reparentTo(base.render2)
    if corresponding_set is not None:
        for pair in corresponding_set:
            correspond_point_src_indx, correspond_point_tgt_indx = pair[0], pair[1]
            point1 = source_temp_np[correspond_point_src_indx]
            point2 = target_temp_np[correspond_point_tgt_indx]
            gm.gen_linesegs([[point1, point2]], thickness=0.0001, rgba=[0, 1, 0, 1])._objpdnp.reparentTo(base.render2)


def global_registration(src, tgt, downsampling_voxelsize=.003, toggledebug=False):
    src_o3d = pda.nparray_to_o3dpcd(src)
    tgt_o3d = pda.nparray_to_o3dpcd(tgt)
    source_down, source_fpfh = __preprocess_point_cloud(src_o3d, downsampling_voxelsize)
    target_down, target_fpfh = __preprocess_point_cloud(tgt_o3d, downsampling_voxelsize)
    distance_threshold = downsampling_voxelsize * 3
    if toggledebug:
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % downsampling_voxelsize)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        print(
            f"   Number of point in source down {len(source_down.points)}, Number of point in source targer {len(target_down.points)}")
    result_global = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.80),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(5000000, 0.99))

    global_trans = result_global.transformation.copy()
    if toggledebug:
        print(f":: Global registeration result: RMSE({result_global.inlier_rmse}), fitness({result_global.fitness})"
              f"len of corresponding set ({len(result_global.correspondence_set)})")
        __draw_registration_result(source_down, target_down, global_trans,
                                   result_global.correspondence_set)

    return global_trans


def __format_validate(src):
    if isinstance(src, np.ndarray):
        src = pda.nparray_to_o3dpcd(src)
    elif isinstance(src, o3d.geometry.PointCloud):
        pass
    else:
        raise Exception("The input format should be numpy array !")
    return src


def _draw_icp_result(source_o3d: o3d.geometry.PointCloud,
                     target_o3d: o3d.geometry.PointCloud,
                     std_out=None) -> None:
    source_tmp = copy.deepcopy(source_o3d)
    target_tmp = copy.deepcopy(target_o3d)
    source_tmp_np = np.asarray(source_tmp.points)
    target_tmp_np = np.asarray(target_tmp.points)
    pcd_source = gm.gen_pointcloud(source_tmp_np, [[1, 0.706, 0, 1]])
    pcd_source.attach_to(std_out)
    pcd_target = gm.gen_pointcloud(target_tmp_np, [[0, 0.651, 0.929, 1]])
    pcd_target.reparentTo(std_out)


def icp(src: np.ndarray,
        tgt: np.ndarray,
        maximum_distance=0.2,
        downsampling_voxelsize=None,
        init_homomat=np.eye(4),
        relative_fitness=1e-11,
        relative_rmse=1e-11,
        max_iteration=9000,
        std_out=None) -> np.ndarray:
    src_o3d = __format_validate(src)
    tgt_o3d = __format_validate(tgt)
    if downsampling_voxelsize is not None:
        src_o3d = src_o3d.voxel_down_sample(downsampling_voxelsize)
        tgt_o3d = tgt_o3d.voxel_down_sample(downsampling_voxelsize)
    if std_out is not None:
        print(":: Point-to-point ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        _draw_icp_result(src_o3d, tgt_o3d)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=relative_fitness,
                                                                 # converge if fitnesss smaller than this
                                                                 relative_rmse=relative_rmse,
                                                                 # converge if rmse smaller than this
                                                                 max_iteration=max_iteration)
    result_icp = o3d.pipelines.registration.registration_icp(src_o3d, tgt_o3d, maximum_distance, init_homomat,
                                                             criteria=criteria)

    return result_icp.transformation.copy()


def locate_rack(rack_pcd, downsampling_voxelsize=0.03, init_rotmat=np.eye(4), toggle=False):
    rack_template = rm.homomat_transform_points(init_rotmat, RACK_HARD_TEMPLATE)
    global_trans = global_registration(src=rack_template,
                                       tgt=rack_pcd,
                                       downsampling_voxelsize=downsampling_voxelsize,
                                       toggledebug=toggle)

    return np.dot(global_trans, init_rotmat)


def evaluate_registration(src, tgt, threshold=0.0005, trans_init=np.eye(4), toggle=True):
    src_tmp = pda.nparray_to_o3dpcd(src)
    tgt_tmp = pda.nparray_to_o3dpcd(tgt)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        src_tmp, tgt_tmp, threshold, trans_init)
    if toggle:
        print(evaluation)
    return evaluation.fitness, evaluation.inlier_rmse


def find_most_clustering_pcd(pcd, nb_distance=0.01, min_points=20):
    pcd_o3d = pda.nparray_to_o3dpcd(pcd)
    labels = np.array(
        pcd_o3d.cluster_dbscan(eps=nb_distance, min_points=min_points, print_progress=False)
    )
    max_label = labels.max()
    return pcd[np.where(labels == max_label)]


def cluster_pcd(pcd, nb_distance=0.01, min_points=20, nb_points=16, radius=0.005, is_remove_outlier=False):
    """
    Cluster the point cloud
    """
    pcd_o3d = pda.nparray_to_o3dpcd(pcd)
    if is_remove_outlier:
        cl, ind = pcd_o3d.remove_radius_outlier(nb_points=nb_points, radius=radius)
        _labels = np.array(
            pcd_o3d.select_by_index(ind).cluster_dbscan(eps=nb_distance, min_points=min_points, print_progress=False)
        )
        labels = -np.ones(len(pcd), dtype=int)
        labels[np.array(ind)] = _labels
    else:
        labels = np.array(
            pcd_o3d.cluster_dbscan(eps=nb_distance, min_points=min_points, print_progress=False)
        )
    return labels


def extc_fgnd_pcd(pcd_fgnd, pcd_bgnd, diff_threshold=.01, ):
    """
    Extact foreground pcd by subtracting a template background pcd
    :param pcd_fgnd:
    :param pcd_bgnd:
    :param diff_threshold:
    :param down_sample:
    :return: index of foreground pcd
    """
    pcd_fgnd_o3d = pda.nparray_to_o3dpcd(pcd_fgnd)
    pcd_bgnd_o3d = pda.nparray_to_o3dpcd(pcd_bgnd)
    dists = pcd_fgnd_o3d.compute_point_cloud_distance(pcd_bgnd_o3d)
    dists = np.asarray(dists)
    ind = np.where(dists > diff_threshold)[0]
    return ind