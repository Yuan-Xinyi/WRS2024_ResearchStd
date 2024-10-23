import numpy as np

import wrs.basis.robot_math as rm
import wrs.vision.depth_camera.util_functions as dcuf
from wrs.basis.trimesh import Trimesh, bounds
import wrs.modeling.geometric_model as gm
import open3d as o3d
import wrs.vision.depth_camera.pcd_data_adapter as pda
from pnt_utils import _draw_icp_result


def __format_validate(src):
    if isinstance(src, np.ndarray):
        src = pda.nparray_to_o3dpcd(src)
    elif isinstance(src, o3d.geometry.PointCloud):
        pass
    else:
        raise Exception("The input format should be numpy array !")
    return src


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


def find_most_clustering_pcd(pcd, nb_distance=0.01, min_points=20):
    pcd_o3d = pda.nparray_to_o3dpcd(pcd)
    labels = np.array(
        pcd_o3d.cluster_dbscan(eps=nb_distance, min_points=min_points, print_progress=False)
    )
    max_label = labels.max()

    print(max_label)
    return pcd[np.where(labels == max_label)]


def oriented_box_icp(pcd: np.ndarray,
                     pcd_template: np.ndarray,
                     downsampling_voxelsize=0.001,
                     std_out=None,
                     toggle_debug=False, ) -> np.ndarray:
    # calculate the oriented bounding box (OBB)
    pcd_inl = dcuf.remove_outlier(src_nparray=pcd.copy(),
                                  downsampling_voxelsize=downsampling_voxelsize,
                                  radius=downsampling_voxelsize * 2)

    pcd_inl = find_most_clustering_pcd(pcd_inl)

    pcd_trimesh = Trimesh(vertices=pcd_inl)
    orient_inv, extent = bounds.oriented_bounds(obj=pcd_trimesh)
    orient = np.linalg.inv(orient_inv)
    init_homo = np.asarray(orient).copy()

    if extent[0] < extent[1]:
        init_homo[:3, :3] = rm.rotmat_from_axangle(axis=init_homo[:3, 2],
                                                   angle=np.deg2rad(90)).dot(init_homo[:3, :3])
    #     print("?")

    # gm.gen_frame(init_homo[:3, 3], init_homo[:3, :3], ).attach_to(base)
    # process the rack to make x,y,z axes face to same direction always
    # z_sim = init_homo[:3, :3].T.dot(np.array([0, 0, 1]))
    # z_ind = np.argmax(abs(z_sim))
    # z_d = np.sign(z_sim[z_ind]) * init_homo[:3, z_ind]
    #
    # x_sim = init_homo[:3, :3].T.dot(np.array([1, 0, 0]))
    # x_ind = np.argmax(abs(x_sim))
    # x_d = np.sign(x_sim[x_ind]) * init_homo[:3, x_ind]
    #
    # y_d = np.cross(z_d, x_d)

    # init_homo[:3, :3] = np.array([x_d, y_d, z_d]).T

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
        obb_gm = gm.GeometricModel(initor=pcd_trimesh.bounding_box_oriented)
        # std_out.attach(node=obb_gm, name="rack obb gm")
        obb_gm.set_rgba([0, 0, 1, .3])
        # obb_gm.attach_to(base)
        gm.gen_frame(pos=init_homo[:3, 3],
                     rotmat=init_homo[:3, :3], length=.8).attach_to(obb_gm)
        # gm.gen_pointcloud(rm.homomat_transform_points(init_homo, pcd_template), rgbas=[[0,0,1,1]], pntsize=5).attach_to(base)

    template_pcd = rm.transform_points_by_homomat(homomat=init_homo,
                                                  points=pcd_template)
    transform = icp(src=template_pcd, tgt=pcd, maximum_distance=.1)
    # transform = transform

    # gm.gen_pointcloud(rm.homomat_transform_points(transform, template_pcd), rgbas=[[1, 0, 0, .3]]).attach_to(base)
    return np.dot(transform, init_homo)


def extract_pcd_by_range(pcd, x_range=None, y_range=None, z_range=None, origin_pos=np.zeros(3), origin_rot=np.eye(3),
                         toggle_debug=False):
    origin_frame = rm.homomat_from_posrot(origin_pos, origin_rot)
    pcd_align = rm.transform_points_by_homomat(np.linalg.inv(origin_frame), pcd)
    pcd_ind = np.ones(len(pcd_align), dtype=bool)
    if x_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 0] >= x_range[0]) & (pcd_align[:, 0] <= x_range[1])
    if y_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 1] >= y_range[0]) & (pcd_align[:, 1] <= y_range[1])
    if z_range is not None:
        pcd_ind = pcd_ind & (pcd_align[:, 2] >= z_range[0]) & (pcd_align[:, 2] <= z_range[1])
    if toggle_debug:
        from wrs.basis.trimesh.primitives import Box
        ext_pcd = pcd_align[np.where(pcd_ind)[0]]
        x_range = x_range if x_range is not None else [ext_pcd[:, 0].min(), ext_pcd[:, 0].max()]
        y_range = y_range if y_range is not None else [ext_pcd[:, 1].min(), ext_pcd[:, 1].max()]
        z_range = z_range if z_range is not None else [ext_pcd[:, 2].min(), ext_pcd[:, 2].max()]
        extract_region = Box(
            box_extents=[(x_range[1] - x_range[0]), (y_range[1] - y_range[0]), (z_range[1] - z_range[0]), ],
            box_center=[(x_range[1] + x_range[0]) / 2, (y_range[1] + y_range[0]) / 2, (z_range[1] + z_range[0]) / 2, ])
        bx_gm = gm.GeometricModel(extract_region)
        bx_gm.rgba = [1, 0, 0, .3]
        bx_gm.homomat = origin_frame
        bx_gm.attach_to(base)
    return np.where(pcd_ind)[0]


if __name__ == "__main__":
    from shapely.geometry import Polygon
    from wrs.basis.trimesh.creation import extrude_polygon
    import wrs.modeling.collision_model as cm
    import pickle


    def rectangle_polygon(rect_center, rect_extent):
        """Generate a rectangle (shapely.geometry.Polygon)"""
        lu = rect_center - rect_extent / 2
        ru = rect_center + np.array([rect_extent[0], -rect_extent[1]]) / 2
        lt = rect_center + np.array([-rect_extent[0], +rect_extent[1]]) / 2
        rt = rect_center + rect_extent / 2
        return Polygon([lu, ru, lt, rt]).convex_hull


    # shape_large = (122 / 1000, 86 / 1000)

    # purple rack
    # shape_large = (116 / 1000, 80 / 1000)
    shape_large = (80 / 1000, 116 / 1000)
    # shape = rectangle_polygon(np.array([0, 0, ]), rect_extent=np.array([7.9 / 100, 11.5 / 100]))
    # shape_in = rectangle_polygon(np.array([0, 0, ]), rect_extent=np.array([6.9 / 100, 10.5 / 100]))
    shape = rectangle_polygon(np.array([0, 0, ]), rect_extent=np.array(shape_large))
    # shape_in = rectangle_polygon(np.array([0, 0, ]),
    #                              rect_extent=np.array([shape_large[0] - 1 / 100, shape_large[1] - 1 / 100]))
    # shape = shape.difference(shape_in)
    rack_mdl = cm.CollisionModel(extrude_polygon(shape, .1 / 1000))
    pcd = rack_mdl.sample_surface(1 / 1000, n_samples=10000)[0]
    print(pcd)
    with open("rack_pcd.pkl", "wb") as f:
        pickle.dump(pcd, f)
