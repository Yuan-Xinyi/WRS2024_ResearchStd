import numpy as np

# from huri.core.common_import import *
import wrs.vision.depth_camera.util_functions as dcuf
from wrs.basis.trimesh import Trimesh, bounds
import pnt_utils as pntu
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial import ConvexHull

import wrs.modeling.geometric_model as gm
import wrs.basis.robot_math as rm


def oriented_box_icp(pcd: np.ndarray,
                     pcd_template: np.ndarray,
                     downsampling_voxelsize=0.006,
                     std_out=None,
                     toggle_debug=False,
                     toggle_remove_pcd_statistical=False,
                     maximum_distance=.1) -> np.ndarray:
    # calculate the oriented bounding box (OBB)
    pcd_inl = dcuf.remove_outlier(src_nparray=pcd.copy(),
                                  downsampling_voxelsize=downsampling_voxelsize,
                                  nb_points=10,
                                  radius=downsampling_voxelsize * 1.5)
    if toggle_remove_pcd_statistical:
        pcd_inl, _ = remove_pcd_statistical(pcd_inl, std_ratio=1)
    pcd_trimesh = Trimesh(vertices=pcd_inl)
    orient_inv, extent = bounds.oriented_bounds(obj=pcd_trimesh)
    orient = np.linalg.inv(orient_inv)
    init_homo = np.asarray(orient).copy()

    if extent[0] >= extent[1]:
        # x is long edge
        long_edge_ind = 0
    else:
        # y is long edge
        long_edge_ind = 1
    # if extent[0] > extent[1]:
    #     init_homo[:3, :3] = rm.rotmat_from_axangle(axis=init_homo[:3, 2],
    #                                                angle=np.deg2rad(90)).dot(init_homo[:3, :3])
    #     print("?")

    z_sim = init_homo[:3, :3].T.dot(np.array([0, 0, 1]))
    z_ind = np.argmax(abs(z_sim))
    z_d = np.sign(z_sim[z_ind]) * init_homo[:3, z_ind]
    x_d = init_homo[:3, long_edge_ind]

    y_d = np.cross(z_d, x_d)

    init_homo[:3, :3] = np.array([x_d, y_d, z_d]).T

    if toggle_debug:
        gm.gen_pointcloud(pcd_inl).attach_to(base)
        gm.gen_frame(init_homo[:3, 3], init_homo[:3, :3]).attach_to(base)

    # gm.gen_frame(init_homo[:3, 3], init_homo[:3, :3], ).attach_to(base)
    # process the rack to make x,y,z axes face to same direction always
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    if init_homo[:3, 2].dot(z_axis) >= 0:
        # rm.angle_between_vectors(init_homo[:3, 2], z_axis)
        pass
    else:
        init_homo[:3, :3] = rm.rotmat_from_axangle(axis=init_homo[:3, 0],
                                                   angle=np.deg2rad(180)).dot(init_homo[:3, :3])
    if init_homo[:3, 1].dot(y_axis) < 0:
        init_homo[:3, :3] = rm.rotmat_from_axangle(axis=init_homo[:3, 2],
                                                   angle=np.deg2rad(180)).dot(init_homo[:3, :3])

    # draw the bounding box
    if std_out is not None or toggle_debug:
        obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounds)
        # std_out.attach(node=obb_gm, name="rack obb gm")
        obb_gm.rgba=[0, 0, 1, .3]
        obb_gm.attach_to(base)
        gm.gen_frame(pos=orient[:3, 3],
                     rotmat=orient[:3, :3]).attach_to(obb_gm)

    template_pcd = rm.transform_points_by_homomat(homomat=init_homo,
                                               points=pcd_template)
    transform = pntu.icp(src=pcd, tgt=template_pcd, maximum_distance=maximum_distance)
    transform = np.linalg.inv(transform)

    # gm.gen_pointcloud(rm.transform_points_by_homomat()(transform, template_pcd), rgbas=[[1, 0, 0, .3]]).attach_to(base)
    return np.dot(transform, init_homo)


def pcd2pcd_icp_match(pcd_1, pcd_2, pcd_1_init_mat):
    """
    Match pcd_1 1 to pcd_2
    :param pcd_1:
    :param pcd_2:
    :return:
    """
    template_pcd = rm.transform_points_by_homomat(homomat=pcd_1_init_mat,
                                               points=pcd_1)
    transform = pntu.icp(src=pcd_2, tgt=template_pcd, maximum_distance=.005,
                         relative_rmse=1e-6, relative_fitness=1e-6, max_iteration=2000)
    transform = np.linalg.inv(transform)
    return np.dot(transform, pcd_1_init_mat)


def oriented_box_icp_general(pcd: np.ndarray,
                             point_cloud_template,
                             downsampling_voxelsize=0.006,
                             std_out=None) -> np.ndarray:
    # calculate the oriented bounding box (OBB)
    pcd_trimesh = Trimesh(vertices=dcuf.remove_outlier(src_nparray=pcd.copy(),
                                                       downsampling_voxelsize=downsampling_voxelsize,
                                                       radius=downsampling_voxelsize * 2))
    orient_inv, extent = bounds.oriented_bounds(mesh=pcd_trimesh)
    orient = np.linalg.inv(orient_inv)
    init_homo = np.asarray(orient).copy()
    if extent[0] > extent[1]:
        init_homo[:3, :3] = rm.rotmat_from_axangle(axis=init_homo[:3, 2],
                                                   angle=np.deg2rad(90)).dot(init_homo[:3, :3])
    # draw the bounding box
    if std_out is not None:
        obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
        # std_out.attach(node=obb_gm, name="rack obb gm")
        obb_gm.set_rgba([0, 0, 1, .3])
        obb_gm.attach_to(base)
        gm.gen_frame(pos=orient[:3, 3],
                     rotmat=orient[:3, :3]).attach_to(obb_gm)
    template_pcd = rm.transform_points_by_homomat(homomat=init_homo,
                                               points=point_cloud_template)
    transform = pntu.icp(src=pcd, tgt=template_pcd, maximum_distance=0.007)
    transform = np.linalg.inv(transform)
    return np.dot(transform, init_homo)


def extrack_rack(pcd: np.ndarray,
                 rack_pcd_template: np.ndarray,
                 results: Dict[str, List],
                 img_shape: np.ndarray,
                 downsampling_voxelsize=0.003,
                 height_lower=.02 + .02,  # .01
                 height_upper=.05 + .015,  # .03
                 std_out=None) -> Tuple[Optional[np.ndarray],
                                        Optional[np.ndarray],
                                        Optional[np.ndarray]]:
    # find the test tube rack label. TODO: consider multiple test tube racks, TODO: Programs to check height lower/upper bounds
    # if there are multiple rack in the results
    if len(results) == 0:
        return None, None, None
    racks_in_img = results[results[:, -1].argsort()[::-1]]
    is_rack_found = False
    for rack_tmp in racks_in_img:
        lt_pos, rb_pos = rack_tmp[1:3], rack_tmp[3:5]
        pcd_yolo = pntu.extract_pcd_by_yolo(pcd=pcd,
                                            img_shape=img_shape,
                                            bound_lt=lt_pos,
                                            bound_rb=rb_pos,
                                            enlarge_detection_ratio=.2)
        pcd_rack = pcd_yolo[np.where((pcd_yolo[:, 2] > height_lower) & (pcd_yolo[:, 2] < height_upper))]
        pcd_yolo_outliner = pcd_yolo[(pcd_yolo[:, 2] > height_upper)]
        # uncomment for debug ###############################
        # gm.gen_pointcloud(pcd_rack).attach_to(base)
        # base.run()
        #######################################################
        # locate the rack using oriented bounding box and icp
        if len(pcd_rack) < 20:
            continue
        rack_transform = oriented_box_icp(pcd=pcd_rack,
                                          pcd_template=rack_pcd_template,
                                          downsampling_voxelsize=downsampling_voxelsize,
                                          std_out=std_out)
        print("rack trans", rack_transform[:3, 3])
        # process the rack to make x,y,z axes face to same direction always
        # y_axis = np.array([0, 1, 0])
        # z_axis = np.array([0, 0, 1])
        rack_height = .055
        rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
        # if rack_transform[:3, 2].dot(z_axis) >= 0:
        #     rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
        # else:
        #     rack_transform[:3, :3] = rm.rotmat_from_axangle(axis=rack_transform[:3, 0],
        #                                                     angle=np.deg2rad(180)).dot(rack_transform[:3, :3])
        # rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
        # if rack_transform[:3, 1].dot(y_axis) < 0:
        #     rack_transform[:3, :3] = rm.rotmat_from_axangle(axis=rack_transform[:3, 2],
        #                                                     angle=np.deg2rad(180)).dot(rack_transform[:3, :3])
        is_rack_found = True
        break
    if is_rack_found:
        return rack_transform, pcd_rack, pcd_yolo_outliner
    else:
        return None, None, None


def extract_points_in_hull(points: np.ndarray,
                           hull: Union[np.ndarray, ConvexHull]) -> np.ndarray:
    if not isinstance(hull, ConvexHull):
        hull = ConvexHull(hull, qhull_options="QJ")
    # TODO: check if this can work
    return points[np.max(np.dot(hull.equations[:, :-1], points.T).T + hull.equations[:, -1], axis=-1) <= 0]


def tube_rm_outlier(pcd_tube: np.ndarray,
                    downsampling_voxelsize: float) -> np.ndarray:
    pcd_tube_rm_outlier, _ = dcuf.remove_outlier(src_nparray=pcd_tube,
                                                 downsampling_voxelsize=downsampling_voxelsize,
                                                 nb_points=5, radius=downsampling_voxelsize * 1.5)
    if len(pcd_tube_rm_outlier) < 10:
        return None
    return extract_points_in_hull(points=pcd_tube, hull=pcd_tube_rm_outlier)


def extrack_tube(pcd: np.ndarray,
                 lt_pos: List,
                 rb_pos: List,
                 img_shape: np.ndarray) -> Optional[np.ndarray]:
    # find the test tube rack label. TODO: consider multiple test tube racks
    pcd_yolo = pntu.extract_pcd_by_yolo(pcd=pcd,
                                        img_shape=img_shape,
                                        bound_lt=lt_pos,
                                        bound_rb=rb_pos,
                                        enlarge_detection_ratio=0)
    return pcd_yolo


def remove_pcd_statistical(src_nparray, nb_neighbors=20, std_ratio=2.):
    src_o3d = pntu.pda.nparray_to_o3dpcd(src_nparray)
    cl, ind = src_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                 std_ratio=std_ratio)
    return pntu.pda.o3dpcd_to_nparray(cl), ind
