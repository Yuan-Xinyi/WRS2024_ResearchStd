import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as gm


def add_task(task, args: list = None, timestep: float = 0.1):
    """
    Add a task to the taskMgr. The name of the function will be the name in the taskMgr
    :param task: a function added to the taskMgr
    :param args: the arguments of function
    :param timestep: time step in the taskMgr
    """
    if args is not None:
        base.taskMgr.doMethodLater(timestep, task, task.__code__.co_name,
                                   extraArgs=args,
                                   appendTask=True)
    else:
        base.taskMgr.doMethodLater(timestep, task, task.__code__.co_name)


def test(pcd, base):
    # Transform and Plot Point Clouds

    pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
    pcd_node = [gm.gen_pointcloud(pcd_r, color_c4)]
    pcd_node[0].attach_to(base)
    angle_resolution = .1
    move_resolution = .0005

    gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                 np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])

    def move_pos(mat, direction):
        mat = mat.copy()
        mat[:3, 3] = mat[:3, 3] + direction
        return mat

    def move_rot(mat, direction, angle):
        mat = mat.copy()
        mat[:3, :3] = np.dot(rm.rotmat_from_axangle(direction, angle), mat[:3, :3])
        return mat

    def adjust_pcd(pcd_node_node, pcd, task):
        if base.inputmgr.keymap["a"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([0, -move_resolution, 0])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(w2cam_homomat[:3, 3], w2cam_homomat[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["d"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([0, move_resolution, 0])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(w2cam_homomat[:3, 3], w2cam_homomat[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["w"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([-move_resolution, 0, 0])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["s"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([move_resolution, 0, 0])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["q"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([0, 0, -move_resolution])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["e"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_pos(tcp_homomat.dot(affine_mat_node[0]),
                                                 np.array([0, 0, move_resolution])))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["z"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([1, 0, 0]),
                                                 np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["x"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([1, 0, 0]),
                                                 -np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["c"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([0, 1, 0]),
                                                 np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["v"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([0, 1, 0]),
                                                 -np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["b"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([0, 0, 1]),
                                                 np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])
        elif base.inputmgr.keymap["n"]:
            affine_mat_node[0] = np.dot(np.linalg.inv(tcp_homomat),
                                        move_rot(tcp_homomat.dot(affine_mat_node[0]), np.array([0, 0, 1]),
                                                 -np.radians(angle_resolution)))
            if pcd_node_node[0] is not None:
                pcd_node_node[0].remove()
            pcd_r = rm.transform_points_by_homomat(tcp_homomat.dot(affine_mat_node[0]), points=pcd)
            pcd_node_node[0] = gm.gen_pointcloud(pcd_r, color_c4)
            pcd_node_node[0].attach_to(base)
            print("np."+repr(affine_mat_node[0]))
            gm.gen_frame(np.dot(tcp_homomat, affine_mat_node[0])[:3, 3],
                         np.dot(tcp_homomat, affine_mat_node[0])[:3, :3]).attach_to(pcd_node[0])

        return task.again

    add_task(adjust_pcd, args=[pcd_node, pcd])

    base.run()


def cobottapos2wrspos(cobbota_pos):
    pos = cobbota_pos[:3]
    rpy = cobbota_pos[3:6]
    return pos, rm.rotmat_from_euler(*rpy)


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
    from wrs.robot_sim.robots.cobotta.cobotta_ripps import CobottaRIPPS
    import cobotta2 as cbtx
    import config_file as conf
    # Init base
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0], lens_type="perspective")  # , lens_type="orthographic"
    # base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])

    rs_pipeline = RealSenseD405()

    tcp2cam_mat = conf.hand2cam_mat

    robot_s = CobottaRIPPS()
    frame_bottom = gm.GeometricModel("./meshes/table_plate.stl")
    frame_bottom.rgba=[.83, .74, .44, 1]
    frame_bottom.pos=np.array([0,0,-0.033])
    frame_bottom.attach_to(base)
    stand = gm.GeometricModel("./meshes/cobotta_stand.stl")
    # stand.attach_to(base)
    stand.pos=np.array([0, 0, -.035])
    stand.rgba=[.55, .55, .55, 1]

    robot_x = cbtx.CobottaX()


    robot_s.gen_meshmodel().attach_to(base)
    pos, rot = cobottapos2wrspos(robot_x.get_pose_values())
    tcp_homomat = rm.homomat_from_posrot(pos, rot)

    w2cam_homomat = np.dot(tcp_homomat, tcp2cam_mat)

    print(tcp_homomat)
    # print(robot_s.get_gl_tcp("arm"))

    pcd, pcd_color, depth_img, color_img = rs_pipeline.req_data()
    pcd, pcd_color, depth_img, color_img = rs_pipeline.req_data()
    pcd, pcd_color, depth_img, color_img = rs_pipeline.req_data()
    pcd, pcd_color, depth_img, color_img = rs_pipeline.req_data()
    pcd, pcd_color, depth_img, color_img = rs_pipeline.req_data()
    color_c4 = np.ones((len(pcd_color), 4))
    color_c4[:, :3] = pcd_color
    print(color_c4)
    # gm.gen_pointcloud(pcd, color_c4).attach_to(base)

    affine_mat_node = [tcp2cam_mat]
    test(pcd, base)
