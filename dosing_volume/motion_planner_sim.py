import copy
import math
import random
import time
import torch
import torchvision
import numpy as np

# import animation_cobotta as ac
import wrs.basis.robot_math as rm
import cobotta2 as cbtx
import env_bulid_kobe as eb
# import fisheye_camera as camera
# import wrs.manipulation.approach_depart_planner as adp
import wrs.modeling.geometric_model as gm
import wrs.motion.probabilistic.rrt_connect as rrtc
import model_loader as model
import wrs.robot_sim.robots.cobotta.cobotta as cbts
import wrs.visualization.panda.world as wd
import config_file as conf

ZERO_ERROR = 0.035 - 0.0048


class MotionPlannerRT():
    def __init__(self, robot_s, robot_x, obstacle_list=[], init_height = 0.01,component_name="arm"):
        self.robot_s = robot_s
        self.robot_x = robot_x
        self.obstacle_list = obstacle_list
        self.init_height = init_height
        self.component_name = component_name
        self.rrtc_planner = rrtc.RRTConnect(self.robot_s)
        # self.ad_planner = adp.ADPlanner(self.robot_s)
        self.create_angle_list()

    def get_rbt_pose_from_pipette(self, pipette_gl_pos, pipette_gl_angle=0, dist=0.007):
        tcp_gl_mat = self.get_tcp_gl_mat_from_pipette(pipette_gl_pos, pipette_gl_angle, dist)
        rbt_tcp_pos = np.array([0, 4.7, 10]) / 1000
        rbt_tcp_mat = rm.homomat_from_posrot(rbt_tcp_pos, np.eye(3))
        rbt_gl_mat = np.dot(tcp_gl_mat, rbt_tcp_mat)
        rot_euler = rm.rotmat_to_euler(rbt_gl_mat[:3, :3])
        return np.append(np.append(rbt_gl_mat[:3, 3], rot_euler), 5)

    def get_tcp_gl_mat_from_pipette(self, pipette_gl_pos, pipette_gl_angle=0, dist=0.007):
        pipette_gl_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(pipette_gl_angle))
        pipette_gl_mat = rm.homomat_from_posrot(pipette_gl_pos, pipette_gl_rot)
        pipette_tcp_pos = np.array([-0.008, -0.15485-self.init_height, 0.01075]) + np.array(
            [0.0015, -dist, -0.0058])  # [y,z,x] in global
        pipette_tcp_rot = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                 rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 2))
        pipette_tcp_mat = rm.homomat_from_posrot(pipette_tcp_pos, pipette_tcp_rot)
        tcp_gl_mat = np.dot(pipette_gl_mat, np.linalg.inv(pipette_tcp_mat))
        return tcp_gl_mat

    def create_angle_list(self):
        angle_half = np.array(range(0, 185, 5))
        angle_list = np.array([0] * 73)
        angle_list[::2] = angle_half
        angle_list[1:][::2] = angle_half[1:] * -1
        # angle_list = angle_list - np.ones(len(angle_list)) * 90
        self.angle_list = list(angle_list)

    def get_tgt_pose(self, tgt_pos):
        # dist = 0.058
        dist = conf.height_rack - conf.base_height
        current_jnts = self.robot_x.get_jnt_values()
        self.robot_s.fk(jnt_values=current_jnts)

        for angle in self.angle_list:
            sample_mat = self.get_tcp_gl_mat_from_pipette(tgt_pos, angle, dist)
            # gm.gen_frame(sample_mat[:3, 3], sample_mat[:3, :3]).attach_to(base)
            tgt_pose = self.get_rbt_pose_from_pipette(tgt_pos, angle, dist)
            # base.run()
            return tgt_pose

    def move_planner(self, tgt_pos, direct_pose=False):
        # dist = 0.058
        dist = conf.height_rack - conf.base_height
        current_jnts = self.robot_x.get_jnt_values()

        for angle in self.angle_list:
            if direct_pose:
                tgt_pose = np.array([0, 0, dist, np.pi / 2, 0, np.pi / 2, 5])
                tgt_pose[:3] += tgt_pos[:3]
                tgt_pose[5] += np.radians(angle)
            else:
                sample_mat = self.get_tcp_gl_mat_from_pipette(tgt_pos, angle, dist)
                # gm.gen_frame(sample_mat[:3, 3], sample_mat[:3, :3]).attach_to(base)
                tgt_pose = self.get_rbt_pose_from_pipette(tgt_pos, angle, dist)
                # base.run()
            try:
                print("tgt_pose:", tgt_pose)
                tgt_jnts = self.robot_x.P2J(tgt_pose)
                if not self.robot_s.is_jnt_values_in_ranges("arm", tgt_jnts):
                    print("out of range")
                    continue
            except:
                # print(tgt_pose)
                # gm.gen_frame(tgt_pose[:3]).attach_to(base)
                self.robot_x.clear_error()
                continue
            # print(f"mat:{sample_mat}")
            if tgt_jnts is not None:
                print(tgt_jnts)
                self.robot_s.fk(jnt_values=tgt_jnts)
                if self.robot_s.is_collided():
                    print("robot collision")
                    continue
                # print(tgt_jnts)
                path = self.rrtc_planner.plan(component_name=self.component_name,
                                              start_conf=current_jnts,
                                              goal_conf=tgt_jnts,
                                              ext_dist=0.2,
                                              max_iter=1000,
                                              obstacle_list=self.obstacle_list,
                                              smoothing_iterations=100)
                # angle_output.append(angle)
                # robot_s.gen_meshmodel().attach_to(base)
                # base.run()
                return path

        print("No result")

        base.run()
        return None




def is_inside_range(jnt_values):
    for i in range(6):
        if jnt_values[i] < robot_s.manipulator.jlc.jnts[i + 1]['motion_rng'][0] or jnt_values[i] > \
                robot_s.manipulator.jlc.jnts[i + 1]['motion_rng'][1]:
            print(jnt_values[i], robot_s.manipulator.jlc.jnts[i]['motion_rng'])
            print(f"{i} out of range")
            # robot_s.fk(jnt_values=jnt_values)
            # robot_s.gen_meshmodel().attach_to(base)
            # base.run()
            return False
    return True




if __name__ == '__main__':

    ## test for the most adapted angle
    angle_half = np.array(range(0, 95, 5))
    angle_list = np.array([0] * 37)
    angle_list[::2] = angle_half
    angle_list[1:][::2] = angle_half[1:] * -1
    # angle_list = angle_list - np.ones(len(angle_list)) * 90
    angle_list = list(angle_list)
    angle_list = np.arange(-90, 95, 5)
    print(angle_list)
    print(len(angle_list))
    spiral_list = rm.gen_3d_equilateral_verts(np.zeros(3), np.eye(3), 4, 0.0009)
    spiral_dist = np.linalg.norm(spiral_list, axis=1)
    # print(spiral_dist)
    '''
    set up    
    '''
    base = wd.World(cam_pos=[0.2445, 0.07175, 0.67275], lookat_pos=[0.2445, 0.07175, 0])
    gm.gen_frame().attach_to(base)

    component_name = "arm"
    manipulator_name = 'hnd'
    robot_s = cbts.Cobotta()
    robot_x = cbtx.CobottaX()
    print("connected")


    env = eb.Env(robot_s)
    mplan = MotionPlannerRT(robot_s, obstacle_list=[env.box[0], env.box[1], env.tip_cm])
    rrtc_planner = rrtc.RRTConnect(robot_s)

    eject_jnt_values = np.array([1.12550394, 0.49365067, 1.48923316, 1.45705496, -1.3587844, -0.22111282])


    time.sleep(1)
