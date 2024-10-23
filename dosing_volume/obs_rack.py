from typing import Union
import pickle
import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as gm

from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
import extract
import icp_extract
import config_file as conf
import itertools

# small
# SHAPE_DIM = (116 / 1000, 80 / 1000)
SHAPE_DIM = (122 / 1000, 86 / 1000)


# OBS_HEGIHT = conf.OBS_HEIGHT


def cobottapos2wrspos(cobbota_pos):
    pos = cobbota_pos[:3]
    rpy = cobbota_pos[3:6]
    return pos, rm.rotmat_from_euler(*rpy)


def wrshomomat2cobbotapos(wrshomomat):
    pos = wrshomomat[:3, 3]
    rpy = rm.rotmat_to_euler(wrshomomat[:3, :3])
    return np.hstack([pos, rpy, -1])


def get_aligned_pcd_im(rs_pipe, rbtx, toggle=False):
    # rbt_jnt = rbtx.get_jnt_values()
    # rbt_s.fk("arm", rbt_jnt)
    # pos, rot = rbt_s.arm.get_gl_tcp(5, tcp_loc_pos=np.zeros(3), tcp_loc_rotmat=np.eye(3))
    # jnt5_homomat = rm.homomat_from_posrot(pos, rot)
    pos, rot = cobottapos2wrspos(rbtx.get_pose_values())
    tcp_homomat = rm.homomat_from_posrot(pos, rot)

    w2cam_homomat = np.dot(tcp_homomat, conf.hand2cam_mat)
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    pcd_aligned = rm.transform_points_by_homomat(w2cam_homomat, pcd)
    color_c4 = np.ones((len(pcd_color), 4))
    color_c4[:, :3] = pcd_color
    if toggle:
        gm.gen_pointcloud(pcd, color_c4).attach_to(base)
        base.run()
    return pcd_aligned, pcd_color, depth_img, color_img


class Rack_Locator(object):
    """
    Locate the pose of the rack
    """

    def __init__(self, arm_x, arm_s,
                 sensor_handler: RealSenseD405,
                 toggle_debug=False):
        self._arm_x = arm_x
        self._arm_s = arm_s
        self._sensor_hdl = sensor_handler
        # ensure to remove empty frame
        for _ in range(5):
            self._sensor_hdl.get_pcd_texture_depth()

        self.toggle_debug = toggle_debug
        self._track_pose = None

        self._obs_p_list = [rm.homomat_from_posrot(pos, rot) for pos, rot in
                            itertools.product(conf.init_obs_pos_list, conf.init_obs_rot_list)]

        with open("rack_pcd.pkl", "rb") as f:
            self._rack_pcd = pickle.load(f)

        if toggle_debug:
            gm.gen_pointcloud(self._rack_pcd).attach_to(base)

    def scan(self, toggle=False):
        pcd_rgba_region = None
        pcds = []
        imgs = []
        for i in range(len(self._obs_p_list)):
            obs_p = self._obs_p_list[i]
            self._arm_x.move_p(pos=obs_p[:3, 3], rot=obs_p[:3, :3])
            pcd_w, pcd_color, _, _ = get_aligned_pcd_im(self._sensor_hdl, self._arm_x, )
            pcd_rgba = np.concatenate((pcd_w, pcd_color), axis=1)
            if toggle:
                rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
                gm.gen_pointcloud(pcd_w[:, :3], rgba=rgba).attach_to(base)
            if i >= 1:
                pcd_rgba_region = np.vstack((pcd_rgba_region, pcd_rgba))
            else:
                pcd_rgba_region = pcd_rgba
            pcds.append(pcd_rgba)
            imgs.append(pcd_color)
        return pcd_rgba_region, pcds, imgs

    def get_pcd_im(self):
        pcd_w, pcd_color, depth_img, color_img = get_aligned_pcd_im(self._sensor_hdl, self._arm_x, )
        return pcd_w, color_img

    def get_pcd_color(self):
        pcd_w, pcd_color, depth_img, color_img = get_aligned_pcd_im(self._sensor_hdl, self._arm_x, )
        return pcd_w, pcd_color

    def locate_rack(self, pcd_region: np.ndarray,
                    rack_height,
                    height_range: Union[tuple, list] = (.055, 0.075),
                    toggle_debug: bool = False, ):
        height_range = np.array(height_range)
        height_range += np.tile(-conf.base_height, 2)
        pcd_region = pcd_region[
            (pcd_region[:, 0] < .5) & (pcd_region[:, 0] > 0) & (pcd_region[:, 1] < .2) & (pcd_region[:, 1] > -.2)]
        pcd_ind = icp_extract.extract_pcd_by_range(pcd=pcd_region[:, :3],
                                                   z_range=height_range,
                                                   toggle_debug=toggle_debug)
        raw_rack_pcd = pcd_region[:, :3][pcd_ind]
        rack_transform = extract.oriented_box_icp(pcd=raw_rack_pcd,
                                                  pcd_template=self._rack_pcd,
                                                  downsampling_voxelsize=.007,
                                                  toggle_debug=toggle_debug)

        if toggle_debug:
            gm.gen_pointcloud(rm.transform_points_by_homomat(rack_transform, self._rack_pcd),
                              rgba=np.array([[1, 0, 0, 1]])).attach_to(base)
            gm.gen_pointcloud(self._rack_pcd).attach_to(base)

        rack_transform[:3, 3] = rack_transform[:3, 3] - rack_transform[:3, 2] * rack_height
        rack_transform[:3, :3] = rack_transform[:3, :3].dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), np.pi))
        return rack_transform

    def obs_in_rack_center(self, rack_tf: np.ndarray, ):
        rack_center_pos = rack_tf[:3, 3]

        angle = rm.angle_between_vectors(np.array([0, 1, 0]), rack_tf[:3, 1])
        rot2rack = rm.rotmat_from_axangle(np.array([0, 0, 1]), angle)
        # euler = rm.rotmat_to_euler()
        cam_obs_pos = np.zeros_like(rack_center_pos)
        cam_obs_pos[:2] = rack_center_pos[:2]
        # w2cam_mat = rm.homomat_from_posrot(*self._arm_x.get_pose()).dot(self._hand_to_eye_mat)
        w2cam_mat_rot = np.array([[0.00187959, -0.99999758, -0.00112828],
                                  [-0.99974083, -0.00190469, 0.02268529],
                                  [-0.02268738, 0.00108535, -0.99974202], ]).T
        w2cam_obs_mat_rot = rot2rack.dot(w2cam_mat_rot)

        cam_obs_pos[2] = conf.OBS_HEIGHT_REFINE
        w2cam_obs_homo = rm.homomat_from_posrot(cam_obs_pos, w2cam_obs_mat_rot)
        w2r_obs_homo = w2cam_obs_homo.dot(np.linalg.inv(conf.hand2cam_mat))
        # euler = np.array([0, 0, 0])
        # suc = self._arm_x.move_pose(wrshomomat2cobbotapos(w2r_obs_homo))

        pcds = []
        pcds_color = []
        rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).T
        rott = np.dot(rm.rotmat_from_axangle([0, 0, 1], -np.radians(30)), np.array(rot))
        rott_euler = rm.rotmat_to_euler(rott)
        for x in np.arange(-51, 51, 50) / 1000:
            for y in np.arange(-43, 44, 80) / 1000:
                try:
                    # print("w2r_obs_homo",w2r_obs_homo[:3, 3])
                    # print("rack_tf_x",rack_tf[:3, 0])
                    jnt = self._arm_x.P2J(wrshomomat2cobbotapos(rm.homomat_from_posrot(
                        w2r_obs_homo[:3, 3] + rack_tf[:3, 0] * x + rack_tf[:3, 1] * y, rott
                    )))
                    # suc = self._arm_x.move_jnts(jnt)
                    pos = w2r_obs_homo[:3, 3] + rack_tf[:3, 0] * x + rack_tf[:3, 1] * y
                    suc = self._arm_x.move_pose(np.hstack([pos, rott_euler, -1]))
                    pcd, pcd_color = self.get_pcd_color()
                    pcds.append(np.concatenate((pcd, pcd_color), axis=1))
                except Exception as e:
                    print(e)
                    # gm.gen_frame(w2r_obs_homo[:3, 3] + rack_tf[:3, 0] * x + rack_tf[:3, 1] * y,
                    #              rott).attach_to(base)
                    self._arm_x.clear_error()

        if len(pcds) == 0:
            return None
        if len(pcds) > 2:
            pcd_rgba_region = np.vstack(pcds)
        else:
            pcd_rgba_region = pcds[0]
        return pcd_rgba_region


def move_to_new_pose(pose, speed=100):
    pose, times = robot_x.null_space_search(pose)
    if pose is not None:
        robot_x.move_pose(pose, speed=speed)
        return times
    else:
        raise Exception("No solution!")


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


def show_pcd_color(rs_pipe, robot_x, robot_s, base):
    pcd_node = [gm.gen_pointcloud([]), robot_s.gen_meshmodel()]

    def show_pcd(pcd_node, task):
        if pcd_node[0] is not None:
            pcd_node[1].detach()
            pcd_node[0].remove()

        robot_s.fk(jnt_values=robot_x.get_jnt_values())
        pcd_aligned, pcd_color, depth_img, color_img = get_aligned_pcd_im(rs_pipe, robot_x)

        # pcd_rgba_ind = icp_extract.extract_pcd_by_range(pcd_aligned, z_range=(-.01, 0.5))
        # pcd_aligned = pcd_aligned[pcd_rgba_ind]
        # pcd_color = pcd_color[pcd_rgba_ind]

        color_c4 = np.ones((len(pcd_color), 4))
        color_c4[:, :3] = pcd_color
        pcd_node[0] = gm.gen_pointcloud(pcd_aligned, color_c4)
        pcd_node[1] = robot_s.gen_meshmodel()
        pcd_node[0].attach_to(base)
        pcd_node[1].attach_to(base)

        return task.again

    add_task(show_pcd, args=[pcd_node])
    base.run()


if __name__ == "__main__":
    import visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD405
    from wrs.robot_sim.robots.cobotta.cobotta_ripps import CobottaRIPPS
    import cobotta2 as cbtx

    import env_bulid_kobe as eb

    # Init base
    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0], lens_type="perspective")  # , lens_type="orthographic"
    # base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])

    rs_pipe = RealSenseD405()

    gm.gen_arrow(spos=np.array([.1, 0, 0]), epos=np.array([.8, 0, 0])).attach_to(base)

    gm.gen_frame(pos=np.array([.1, -.3, 0])).attach_to(base)
    gm.gen_frame(pos=np.array([.1, .3, 0])).attach_to(base)

    robot_s = CobottaRIPPS()
    robot_x = cbtx.CobottaX()

    robot_x.defult_gripper()
    rbt_jnt = robot_x.get_jnt_values()
    # robot_s.gen_meshmodel().attach_to(base)

    env = eb.Env(robot_s)

    # show_pcd_color(rs_pipe,robot_x,robot_s,base)

    # jnt_value = robot_x.P2J([0.25, 0, 0.20, *np.deg2rad(np.array([-90, 90, -85])), -1])
    # print(repr(jnt_value))
    # robot_x.move_jnts(jnt_value)
    #
    # jnt_value = robot_x.P2J([0.33, 0, 0.20, *np.deg2rad(np.array([-90, 90, -85])), -1])
    # print(repr(jnt_value))
    # robot_x.move_jnts(jnt_value)
    #
    # jnt_value = robot_x.P2J([0.33, 0.08, 0.20, *np.deg2rad(np.array([-90, 90, -85])), -1])
    # print(repr(jnt_value))
    # robot_x.move_jnts(jnt_value)
    # #
    # jnt_value = robot_x.P2J([0.25, 0.08, 0.20, *np.deg2rad(np.array([-90, 90, -85])), -1])
    # print(repr(jnt_value))
    # robot_x.move_jnts(jnt_value)
    # robot_s.gen_meshmodel().attach_to(base)

    toggle_debug = True
    toggle_capture = True
    gm.gen_frame().attach_to(base)
    rl = Rack_Locator(robot_x, robot_s, rs_pipe, toggle_debug=toggle_debug)
    # pcd_rgba_region, pcds, imgs = rl.scan(toggle=toggle_debug)
    # pcd_rgba_region, pcds, imgs = rl.scan(toggle=toggle_debug)
    if toggle_capture:
        # robot_x.move_jnts(conf.eject_jnt_values_list[0])
        pcd_rgba_region, pcds, imgs = rl.scan(toggle=False)

        # rack_transform = rl.locate_rack(pcd_rgba_region, rack_height=conf.WORK_RACK_HEIGHT
        #                                 , toggle_debug=False, height_range=conf.WORK_HEIGHT_RANGE)

        import utils_rack_old as utils
        import modeling.collision_model as cm

        # tip_rack = utils.Base96("./meshes/rack_mbp.stl")
        # tip_rack.set_homomat(rack_transform)
        # tip_rack.attach_to(base)
        # tip_rack.show_localframe()
        # base.run()

        # gm.gen_pointcloud(pcd_rgba_region).attach_to(base)
        # pcd_rgba_region = rl.obs_in_rack_center(rack_transform)
        with open("region", "wb") as f:
            pickle.dump(pcd_rgba_region, f)
    else:
        with open("region", "rb") as f:
            pcd_rgba_region = pickle.load(f)
    if pcd_rgba_region is None:
        base.run()

    gm.gen_pointcloud(pcd_rgba_region).attach_to(base)
    rack_transform = rl.locate_rack(pcd_rgba_region, rack_height=conf.WORK_RACK_HEIGHT
                                    , toggle_debug=True, height_range=conf.WORK_RACK_HEIGHT_REFINE)

    # rack_transform[:3, :3] = rm.rotmat_from_axangle(rack_transform[:3, 2], np.radians(90)).dot(rack_transform[:3, :3])
    gm.gen_frame(rack_transform[:3, 3], rack_transform[:3, :3], length=.5).attach_to(base)

    import utils_rack_old as utils
    import modeling.collision_model as cm

    tip_rack = utils.Base96(conf.WORK_RACK_PATH)

    tip_rack.set_homomat(rack_transform)
    tip_rack.attach_to(base)
    tip_rack.show_localframe()
    pcd_transform = rack_transform.copy()
    print(repr(pcd_transform))
    print(pcd_transform[:3, 3])
    pcd_transform[2, 3] = conf.WORK_HEIGHT_RANGE[0]
    print(pcd_transform[:3, 3])
    gm.gen_box(np.array([80 / 1000, 116 / 1000, conf.WORK_HEIGHT_RANGE[1] - conf.WORK_HEIGHT_RANGE[0]]),
               homomat=pcd_transform,
               rgba=[1, 0, 0, .3]).attach_to(base)
    with open("pcd_transform_file", "wb") as f:
        pickle.dump(pcd_transform, f)

    show_pcd_color(rs_pipe, robot_x, robot_s, base)
