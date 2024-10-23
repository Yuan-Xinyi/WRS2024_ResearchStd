import copy
import math
import os.path
import time

import cv2
import numpy as np
import torch

import wrs.basis.robot_math as rm
import cobotta2 as cbtx
import env_bulid_kobe as eb
# import fisheye_camera as camera
import model_loader as model
import wrs.modeling.geometric_model as gm
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.robot_sim.robots.cobotta.cobotta_ripps as cbtr
import wrs.visualization.panda.world as wd
import wrs.visualization.panda.panda3d_utils as p3du
import config_file as conf

import icp_extract
from torchvision import transforms

from realsensecrop import RealSenseD405Crop, letterbox
from sample import gen_2d_isosceles_verts
import file_sys as fs

ZERO_ERROR = 0.035 - 0.0048
speed_general = 70


def num_recog(img, model_num, adjust_data, toggle_debug=False):
    # color_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)[890:1020, 290:585]
    color_img = copy.deepcopy(img)
    wide = 44.5
    height = -0.5
    left = adjust_data[0]
    top = adjust_data[1]
    button = top + 90
    img_separate = []
    img_show = copy.deepcopy(color_img)
    for i in range(5):
        img_num = color_img[int(top + height * i):int(button + height * i),
                  np.floor(left + wide * i).astype(int):np.ceil(left + wide * (i + 1)).astype(int)]
        img_separate.append(img_num)
        cv2.rectangle(img_show, (int(left + wide * i), int(button + height * i)),
                      (int(left + wide * (i + 1)), int(top + height * i)), [60, 40, 80])
    index_list = []
    np.set_printoptions(precision=3, suppress=True)
    for img in img_separate:
        index = model_num.get_score(img)
        index_list.append(index[0])
    print(index_list)
    if toggle_debug:
        cv2.imshow("img_debug", img_show)
        cv2.waitKey(1)
    return index_list


def get_weight_from_str(str_list):
    weight = 0
    for id, num in enumerate(str_list):
        weight += int(num) / (10 ** (id))
    weight = np.around(weight, 4)
    print(weight)
    return weight


def get_pipette_gl_mat_from_rbt_pose(pose_values):
    rbt_gl_pose = pose_values[:3]
    rbt_gl_rot = rm.rotmat_from_euler(np.pi / 2, 0, np.pi / 2)
    tcp_gl_pos = rbt_gl_pose - np.array([0, 4.7, 10]) / 1000
    tcp_gl_rot = rbt_gl_rot
    tcp_gl_mat = rm.homomat_from_posrot(tcp_gl_pos, tcp_gl_rot)
    pipette_tcp_pos = np.array([-0.008, -0.15485, 0.01075]) + np.array([0.0015, -0.058, -0.0058])  # [y,z,x] in global
    pipette_tcp_rot = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                             rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 2))
    pipette_tcp_mat = rm.homomat_from_posrot(pipette_tcp_pos, pipette_tcp_rot)
    pipette_gl_mat = np.dot(tcp_gl_mat, pipette_tcp_mat)
    return pipette_gl_mat


def get_environment_mat(file_name):
    rbt_pos_list = np.loadtxt(file_name)
    tip_pos_list = []
    for pose in rbt_pos_list:
        tip_mat = get_pipette_gl_mat_from_rbt_pose(pose)
        tip_pos = tip_mat[:3, 3]
        tip_pos[2] = 0.035
        # gm.gen_sphere(tip_pos).attach_to(base)
        tip_pos_list.append(tip_pos)
    tip_pos_list = np.array(tip_pos_list)
    pos = tip_pos_list[0]
    x_list = np.vstack([tip_pos_list.T[0], np.ones(len(tip_pos_list))]).T
    y_list = tip_pos_list.T[1]
    m, c = np.linalg.lstsq(x_list, y_list, rcond=None)[0]
    print(m, pos)
    env_rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.arctan(m) + np.pi / 2)
    env_pos = np.array([pos[0], pos[1], 0.003]) + rm.rotmat_from_axangle([0, 0, 1], np.arctan(m)) \
        .dot(np.array([0.009 * 5.5, 0.009 * 3.5, .00]))
    return env_pos, env_rotmat


def get_rbt_mat(file_name):
    rbt_pos_list = np.loadtxt(file_name)
    rbt_pos_list = rbt_pos_list.T[:3].T
    # for pose in rbt_pos_list:
    #     gm.gen_sphere(pose[:3]).attach_to(base)
    pos0 = rbt_pos_list[0]
    x_list = np.vstack([rbt_pos_list.T[0], np.ones(len(rbt_pos_list))]).T
    y_list = rbt_pos_list.T[1]
    m, c = np.linalg.lstsq(x_list, y_list, rcond=None)[0]
    print(m, pos0)
    env_rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.arctan(m) + np.pi / 2)
    env_pos = np.array([pos0[0], pos0[1], 0]) + rm.rotmat_from_axangle([0, 0, 1], np.arctan(m)) \
        .dot(np.array([0.009 * 5.5, 0.009 * 3.5, .00]))
    return env_pos, env_rotmat


def get_aligned_pcd_im(rs_pipe, rbtx, rbt_s, toggle=False):
    jnt5tocam_mat = np.array([[4.14951611e-04, 1.01091359e-02, 9.99948809e-01,
                               5.98077639e-02],
                              [9.99542543e-01, 3.02353296e-02, -7.20452248e-04,
                               -5.97662246e-02],
                              [-3.02410661e-02, 9.99491681e-01, -1.00919644e-02,
                               1.66160307e-02],
                              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                               1.00000000e+00]])
    rbt_jnt = rbtx.get_jnt_values()
    rbt_s.fk(jnt_values=rbt_jnt)
    pos, rot = rbt_s.manipulator.get_gl_tcp(5, tcp_loc_pos=np.zeros(3), tcp_loc_rotmat=np.eye(3))
    jnt5_homomat = rm.homomat_from_posrot(pos, rot)
    w2cam_homomat = np.dot(jnt5_homomat, jnt5tocam_mat)
    pcd, pcd_color, depth_img, color_img = rs_pipe.get_pcd_texture_depth()
    pcd_aligned = rm.homomat_transform_points(w2cam_homomat, pcd)
    color_c4 = np.ones((len(pcd_color), 4))
    color_c4[:, :3] = pcd_color
    if toggle:
        gm.gen_pointcloud(pcd, color_c4).attach_to(base)
    return pcd_aligned


from motion_planner_sim import MotionPlannerRT


def move_to_new_pose(tgt_pose, speed=100):
    if len(tgt_pose) == 3:
        tgt_pose = np.append(tgt_pose, np.array([np.pi / 2, 0, np.pi / 2, -1]))
    if len(tgt_pose) == 6:
        tgt_pose = np.append(tgt_pose, -1)
    pose, times = robot_x.null_space_search(tgt_pose)
    if pose is not None:
        robot_x.move_pose(pose, speed=speed)
        return pose
    else:
        raise Exception("No solution!")


init_xy_array = rm.gen_2d_isosceles_verts(nlevel=4, edge_length=0.0009, nedges=6)

from direct.gui.DirectGui import *
from direct.gui.DirectFrame import DirectFrame
from panda3d.core import (Filename, TextNode,
                          LPoint3f,
                          LVecBase3f,
                          LVecBase4f,
                          TextNode)
from direct.stdpy import threading
import os
import wrs.modeling.geometric_model as gm
from wrs.visualization.panda.panda3d_utils import DrawText


def to_gui_size(tgt_width, img_wh):
    gui_height = img_wh[1] / (img_wh[0] / tgt_width)
    return (tgt_width, 1, gui_height)


class HLabGUI(object):
    """
    the graphical user interface of the application

    author: weiwei
    date: 20180925
    """

    def __init__(self, scenarioctrl=None):
        self.scctrl = scenarioctrl
        this_dir, this_filename = os.path.split(__file__)
        # self.imageObject = OnscreenImage(image="./gui/banner250x1200.png", pos=(1.55, 0, 0), scale=(250 / 1200.0, 1, 1))

        self.pg234 = DirectFrame(
            frameSize=(0, 1, 0, 1),
            frameColor=(1.0, .01, 1.0, 1.0),
            pos=LPoint3f(1.25, 0, 1),
            parent=None,
        )
        self.pg234.setTransparency(0)

        tgt_width = 0.23
        wrs_logo_2022_size = to_gui_size(tgt_width, (1805, 1331))
        self.wrs_logo = OnscreenImage(image="./gui/wrs_logo_2022.png",
                                      pos=(wrs_logo_2022_size[0] - .1, 0, -wrs_logo_2022_size[0]),
                                      scale=wrs_logo_2022_size,
                                      parent=self.pg234)
        self.wrs_logo.setTransparency(1)

        self.riken_u_logo = OnscreenImage(image="./gui/riken_logo.png",
                                          pos=(0, 1, -1 + 176 / 1200 / 2),
                                          scale=(1450 / 1980, 1, 176 / 1980),
                                          parent=None)
        self.riken_u_logo.setTransparency(1)
        left_margin = .1
        brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttonrun_maps.egg"))
        maps = loader.loadModel(brmappath)
        self.run_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttonrun_ready'),
                                                                        maps.find('**/buttonrun_click')),
                                    pos=(0.32 - left_margin, 0, -.5), scale=(.06, .12, .12),
                                    command=self.execplan,
                                    parent=self.pg234)

        # brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttondelete_maps.egg"))
        # maps = loader.loadModel(brmappath)
        # self.del_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttondelete_ready'),
        #                                                                 maps.find('**/buttondelete_click')),
        #                             pos=(0.47 - left_margin, 0, -.5), scale=(.06, .12, .12),
        #                             command=self.deleteCapture,
        #                             parent=self.pg234)

        brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttonrecog_maps.egg"))
        maps = loader.loadModel(brmappath)
        self.recog_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttonrecog_ready'),
                                                                          maps.find('**/buttonrecog_click')),
                                      pos=(0.17 - left_margin, 0, -.5), scale=(.06, .12, .12),
                                      command=self.recognize,
                                      parent=self.pg234)

        self.nposes = 0
        self.textNPose = OnscreenText(text='#Poses: ' + str(self.nposes), pos=(1.45, -.9, 0), scale=0.03,
                                      fg=(1., 1., 1., 1),
                                      align=TextNode.ALeft, mayChange=1)
        self.textCaptured = OnscreenText(text='Ready to capture', pos=(1.45, -.95, 0), scale=0.03, fg=(1., 1., 1., 1),
                                         align=TextNode.ALeft, mayChange=1)

        self.img_screen1 = p3du.ImgOnscreen(size=(225, 225),
                                            pos=(-1 + 1.1 * 400 / 1920, 0, 1 - 1.1 * 225 / 1200),
                                            parent_np=base)
        self.img_screen1.hide()

        i_h, i_w = (176, 400)
        self.img_screen2 = p3du.ImgOnscreen(size=(i_w, i_h),
                                            pos=(-1 + 1.1 * i_w / 1920, 0, 1 - 1.1 * (i_h + 2 * 225) / 1200),
                                            parent_np=base)
        self.img_screen2.hide()

        font_size = .055
        self.tt = DrawText(parent=base.render2d,
                           pos=(-1 + 1.1 * i_w / 1920, 1 - 1.1 * (110 + i_h * 2 + 2 * 225) / 1200, 0),
                           scale=(font_size, font_size * 1920 / 1200))

    def execplan(self):
        self.run_btn['state'] = DGG.DISABLED
        thread = threading.Thread(target=run, args=(self.scctrl,))
        thread.start()
        # thread.join()
        print("thread finished...exiting")

    def deleteCapture(self):
        self.run_btn['state'] = DGG.NORMAL
        self.recog_btn['state'] = DGG.NORMAL
        # self.scctrl.remove_plot()
        self.scctrl.plot_rbt_sync()

    def recognize(self):
        self.recog_btn['state'] = DGG.DISABLED
        self.scctrl.vision(toggle_debug=False, toggle_capture=False)


def run(self, toggle_debug=False):
    if self.rack_transform is None:
        # print("Run vision first")
        self.rack_transform = conf.rack_transform
        self.env.tip_rack.set_homomat(self.rack_transform)
        self.env.tip_rack.attach_to(base)

        self.plot_rbt_sync()
        self.update_tips()
        self.show_all_tips()
        return

    rack_transform = self.rack_transform
    mplan = self.mplan
    robot_x = self.robot_x
    robot_s = self.robot_s
    env = self.env

    tgt_pos = env.tip_rack._hole_pos_list[0]
    if toggle_debug:
        gm.gen_sphere(tgt_pos, radius=0.01).attach_to(base)

    robot_x.defult_gripper()

    step_list = []
    step_arr = np.zeros(96)

    tip_id = 0
    exp_id = 0
    total_num = 0
    total_err = np.zeros(2)

    tip_rack_pos_i = env.tip_rack.get_pos()
    print("rack pose: ", tip_rack_pos_i)

    while tip_id < 96:
        print(tip_id)
        '''
        pick up new tip
        '''
        tgt_pos = env.tip_rack._hole_pos_list[tip_id]
        # tgt_pos[:2] -= total_err  # closed-loop correction

        # draw draw
        tip_rack_pos = tip_rack_pos_i.copy()
        print(tip_rack_pos)
        tip_rack_pos[:2] = tip_rack_pos[:2] - total_err
        print(tip_rack_pos)
        env.tip_rack.set_pos(tip_rack_pos)

        print("tgt_pos:", tgt_pos)
        move_result = False

        # robot_x.move_pose(conf.execute_pose[0])
        # cbtx.open_door(robot_x)
        # robot_x.move_pose(conf.execute_pose[0])

        rbt_tgt_pose = mplan.get_tgt_pose(tgt_pos)
        rbt_tgt_pose[:3] += conf.adjust_pos
        print(env.tip_rack._hole_pos_list)
        print(env.tip_rack.get_pos(),env.tip_rack.get_rotmat())
        breakpoint()
        print(rbt_tgt_pose)
        move_to_new_pose(rbt_tgt_pose)
        # self.plot_rbt_sync()
        # time.sleep(.5)
        current_pose = robot_x.get_pose_values()
        # print("current_pose:", current_pose)
        # time.sleep(0.1)

        current_pose[2] = conf.recognize_height
        new_pose = move_to_new_pose(current_pose)


        # time.sleep(0.5)
        # robot_s.fk(jnt_values=robot_x.get_jnt_values())
        # self.plot_rbt_sync()
        # base.run()

        '''
        adjust position
        '''
        err_id_list = [0]
        err_list = [np.zeros(3)]
        step = 0
        pre = 0
        pre_pre = 0
        pre_pre_pre = 0
        pos_err_total = np.zeros(2, dtype=float)
        # self.plot_rbt_sync()
        while 1:
            # time.sleep(2)
            current_pose = robot_x.get_pose_values()
            time.sleep(0.2)
            pic = rs_pipeline.get_learning_feature()[1]
            time.sleep(0.1)
            pic = rs_pipeline.get_learning_feature()[1]
            time.sleep(0.2)
            resized_frame_row = letterbox(pic, new_shape=[360, 360], auto=False)[0]
            gui.img_screen1.update_img(resized_frame_row)
            gui.img_screen1.show()
            # pic_masked = cv2.bitwise_and(pic, pic, mask=mask)
            with torch.no_grad():
                direct_trans_tip, _, _ = model_vit.get_score(pic)
            score = direct_trans_tip
            print("score", score)
            # pic_resized = letterbox(pic, new_shape=[360, 640], auto=False)[0]
            # cv2.imshow("judge", pic_resized)
            # cv2.imshow("masked",pic_masked)
            # cv2.waitKey(1)

            if score ==0:
                total_err += pos_err_total
                break
            else:
                rotmat = rm.rotmat_from_axangle(np.array([0, 0, 1]), current_pose[5] - math.pi)
                xy_array = gen_2d_isosceles_verts(nlevel=4, edge_length=0.0009, nedges=6)
                print("judge:", pre_pre, pre, score)
                weight = .8
                if score == pre_pre:
                    print("repeated")
                    xy_pose = xy_array[score] * 0.5 * weight
                elif min(score, pre, pre_pre) > 0 and max(score, pre, pre_pre) < 7:
                    xy_pose = xy_array[score] * 0.5 * weight
                else:
                    xy_pose = xy_array[score] * weight

                xyz_pose = np.array([xy_pose[0], xy_pose[1], 0])

                print(xyz_pose)
                pos_err = rotmat.dot(xyz_pose)
                print(pos_err)
                current_pose[:2] -= pos_err[:2]

                pos_err_total += pos_err[:2]
                # time.sleep(1)
                times = robot_x.move_to_pose_nullspace(current_pose)
                # print(times)
                # rotation_times += times
                step += 1
                pre_pre_pre = pre_pre
                pre_pre = pre
                pre = score
                time.sleep(0.2)
            if step > 10:
                tip_id += 1
                step = 0
                tgt_pos = env.tip_rack._hole_pos_list[tip_id]
                # tgt_pos[:2] -= total_err
                rbt_tgt_pose = mplan.get_tgt_pose(tgt_pos)
                rbt_tgt_pose[:3] += conf.adjust_pos
                print(rbt_tgt_pose)
                move_to_new_pose(rbt_tgt_pose)
                # self.plot_rbt_sync()
                # time.sleep(.5)
                current_pose = robot_x.get_pose_values()
                # print("current_pose:", current_pose)
                # time.sleep(0.1)

                current_pose[2] = conf.recognize_height
                new_pose = move_to_new_pose(current_pose)
                # self.plot_rbt_sync()
                continue
        # input(direct_trans_tip)
        # '''
        # insert tip
        # '''
        # current_pose = robot_x.get_pose_values()
        # # input(repr(current_pose))
        # insert_pose = copy.deepcopy(current_pose)
        # insert_pose[2] -= conf.height_insert_tip
        # move_to_new_pose(insert_pose, 70)
        # # self.plot_rbt_sync()
        # # while
        # insert_pose[2] -= 0.0005
        # move_to_new_pose(insert_pose, 20)
        # # self.plot_rbt_sync()
        # try:
        #     move_to_new_pose(insert_pose, 70)
        #     insert_pose[2] -= 0.001
        #     move_to_new_pose(insert_pose, 20)
        # except:
        #     robot_x.clear_error()
        #
        # try:
        #     insert_pose[2] += 0.002
        #     move_to_new_pose(insert_pose, 20)
        # except:
        #     robot_x.clear_error()
        # # input()
        #
        # rise_pose = copy.deepcopy(current_pose)
        # rise_pose[2] -= 0.0075
        # # arise_tip(rise_pose)
        # move_to_new_pose(rise_pose, 20)
        # rise_pose[2] += 0.0075
        # rise_pose = move_to_new_pose(rise_pose, 50)
        # print("phase1")
        # rise_pose[2] += 0.03
        # rise_pose = move_to_new_pose(rise_pose)
        # print("phase2")
        # rise_pose[2] += 0.04
        # rise_pose = move_to_new_pose(rise_pose)
        # print("phase3")


        '''
        get chemical
        '''
        move_result = False
        # chemical_pos = env.deep_plate._hole_pos_list[tip_id]
        chemical_pos = copy.deepcopy(env.microplate._hole_pos_list[exp_id % 24])
        # gm.gen_sphere(chemical_pos).attach_to(base)
        print(chemical_pos)
        chemical_pos[:3] += conf.adjust_pos
        chemical_pos[2] = conf.chemical_height

        move_to_new_pose(chemical_pos)
        time.sleep(0.1)
        current_pose = robot_x.get_pose_values()
        # input("STOP chemical")

        # '''
        # insert chemical
        # '''
        # print("insert chemical")
        # current_pose = robot_x.get_pose_values()
        # time.sleep(0.1)
        # insert_pose = copy.deepcopy(current_pose)
        # robot_x.open_gripper()
        # insert_pose[2] -= conf.height_insert_chem
        # move_to_new_pose(insert_pose, 30)
        # # robot_x.defult_gripper(speed=70)
        # # robot_x.open_gripper()
        # robot_x.defult_gripper(dist=0.018, speed=50)
        # # robot_x.defult_gripper(speed=50)
        # # time.sleep(0.5)
        # move_to_new_pose(current_pose)
        #
        # '''
        # water plant
        # '''
        # print("water plant")
        # robot_x.move_pose(conf.execute_pose[conf.init_id],speed=speed_general)
        #
        # robot_x.move_pose(conf.execute_pose[conf.balance_temp_id],speed=speed_general)
        # balance_pose_rbt = copy.deepcopy(conf.execute_pose[conf.balance_id])
        # balance_pose_rbt[0] -= 0.018 * (exp_id % 4)
        #
        # robot_x.move_pose(balance_pose_rbt,speed=speed_general)
        # balance_pose_rbt[2] -= 0.015
        # robot_x.move_pose(balance_pose_rbt,speed=speed_general)
        # # print(f"position{exp_id % 4} pose: np.{repr(robot_x.get_pose_values())}")
        # # print(f"position{exp_id % 4} jnt_values: np.{repr(robot_x.get_jnt_values())}")
        # robot_x.open_gripper()
        # time.sleep(0.5)
        # robot_x.defult_gripper()
        # balance_pose_rbt[2] += 0.015
        # robot_x.move_pose(balance_pose_rbt,speed=speed_general)
        # robot_x.move_pose(conf.execute_pose[conf.balance_temp_id],speed=speed_general)
        #
        #
        #
        # # time.sleep(1)
        #
        # '''
        # throw away tip
        # '''
        # print("throw away tip")
        #
        # eject_jnt_values = eject_jnt_values_list[tip_id % eject_len]
        # # move_to_new_pose(eject_pose)
        # time.sleep(0.2)
        # robot_x.move_jnts(eject_jnt_values)
        #
        # current_pose = robot_x.get_pose_values()
        # time.sleep(0.1)
        # current_pose[2] -= 0.05
        # move_to_new_pose(current_pose)
        # robot_x.close_gripper()
        # robot_x.defult_gripper()
        #
        # cbtx.close_door(robot_x)
        #
        # """
        # record weight
        # """
        # robot_x.move_jnts(conf.photo_start_jnts_values)
        # robot_x.move_pose(conf.photo_temp_pose_values)
        # robot_x.move_pose(conf.photo_end_pose_values)
        # time.sleep(0.1)
        # # input()
        # # path = rrtc_planner.plan(component_name="arm",
        # #                      start_conf=robot_x.get_jnt_values(),
        # #                      goal_conf=conf.photo_jnt_values,
        # #                      ext_dist=0.2,
        # #                      max_iter=1000,
        # #                      obstacle_list=[env.balance, env.cobotta_stand, env.frame_bottom],
        # #                      smoothing_iterations=100)
        # # print(path)
        #
        # # input("STOP screen")
        # time.sleep(1.5)
        # balance_img = rs_pipeline.get_color_img()
        # time.sleep(0.1)
        # crop_weight = fs.load_json('resources/crop_weight.json')
        # balance_img_rotate = cv2.rotate(balance_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # # color_img = rs_pipeline.crop_img(balance_img_rotate, crop_weight["left_top"], crop_weight["img_size"])
        # color_img = rs_pipeline.crop_img(balance_img_rotate, crop_weight["left_top"], np.floor(np.array(crop_weight["img_size"])*1.034).astype(int))
        # color_img = letterbox(color_img,new_shape=(130,295),auto=False)[0]
        # gui.img_screen2.update_img(color_img)
        # gui.img_screen2.show()
        # # cv2.imwrite(f"{save_path}/{tip_id}.jpg", balance_img)
        # num_list = num_recog(color_img, model_num, crop_weight["left_top_p"], toggle_debug=False)
        # gui.tt.update_text(f"Weight: {num_list[0]}.{num_list[1]}{num_list[2]}{num_list[3]}{num_list[4]}g")
        # gui.tt.show()
        # print(num_list)
        # balance_list.append(num_list)
        # # with open("./data/liquid_weight/balance_num", "wb") as f:
        # #     pickle.dump(balance_list, f)
        # time.sleep(0.5)
        # robot_x.move_pose(conf.photo_temp_pose_values)
        # robot_x.move_jnts(conf.photo_start_jnts_values)
        #
        # # balance_img = cv2.imread("267.jpg")
        # # crop_weight = fs.load_json('resources/crop_weight_test.json')
        # # num_list = num_recog(balance_img, model_num, crop_weight["left_top_p"], toggle_debug=True)
        # # gui.img_screen2.update_img(balance_img)
        # # gui.img_screen2.show()
        # # gui.tt.update_text(f"Weight: {num_list[0]}.{num_list[1]}{num_list[2]}{num_list[3]}{num_list[4]}g")
        # # gui.tt.show()
        # # print(num_list)
        # # balance_list.append(num_list)
        # # with open("./data/liquid_weight/balance_num", "wb") as f:
        # #     pickle.dump(balance_list, f)
        time.sleep(0.5)

        tip_id += 1
        exp_id += 1

        # input(f"tip_id:{tip_id}")

        time.sleep(0.1)


class Execution(object):
    def __init__(self, robot_s, robot_x, rs_pipeline, env, mplan):
        self.component_name = "arm"
        self.manipulator_name = 'hnd',
        self.robot_s = robot_s
        self.robot_x = robot_x
        self.env = env
        self.mplan = mplan

        self.plot_node_rbt = None
        self.plot_node_rack = None

        self.rl = Rack_Locator(robot_x, robot_s, rs_pipeline)
        self.rack_transform = None
        self.plot_rbt_sync()

        self.tips_collections = []
        self.tips_display_toggle = np.ones(96, dtype=bool)
        tip = gm.GeometricModel("./meshes/tip.stl")
        tip.rgba=[200 / 255, 180 / 255, 140 / 255, 1]
        for tip_id in range(96):
            tip_new = copy.deepcopy(tip)
            self.tips_collections.append(tip_new)

    def update_tips(self):
        for tip_id, toggle in enumerate(self.tips_display_toggle):
            if toggle:
                pos_rack = env.tip_rack._hole_pos_list[tip_id]
                self.tips_collections[tip_id].pos=pos_rack

    def show_all_tips(self):
        self.update_tips()
        for tip_id, toggle in enumerate(self.tips_display_toggle):
            if toggle:
                self.tips_collections[tip_id].attach_to(base)

    def unshow_tip(self, tip_id):
        self.tips_display_toggle[tip_id] = False
        self.tips_collections[tip_id].detach()

    def plot_rbt_sync(self):
        self.sync()
        if self.plot_node_rbt is not None:
            self.plot_node_rbt.detach()
        self.plot_node_rbt = self.robot_s.gen_meshmodel()
        self.plot_node_rbt.attach_to(base)

    def vision(self, toggle_debug=False, toggle_capture=True):
        '''
        recognize the rack
        '''
        rl = self.rl
        env = self.env

        move_to_new_pose(conf.start_pose)
        if toggle_capture:
            pcd_rgba_region, pcds, imgs = rl.scan(toggle=toggle_debug)
            # if pcd_rgba_region is not None:
            #     color_c4 = np.ones((len(pcd_rgba_region), 4))
            #     color_c4[:, :3] = pcd_rgba_region[:, 3:]
            #     gm.gen_pointcloud(pcd_rgba_region[:, :3], color_c4).attach_to(base)
            #     input()
            rack_transform = rl.locate_rack(pcd_rgba_region, rack_height=conf.WORK_RACK_HEIGHT
                                            , toggle_debug=toggle_debug, height_range=conf.WORK_HEIGHT_RANGE)
            # pcd_rgba_region = rl.obs_in_rack_center(rack_transform)
            # pcd_rgba_ind = extract.extract_pcd_by_range(pcd_rgba_region, z_range=(-.01, 0.1))
            pcd_rgba_ind = icp_extract.extract_pcd_by_range(pcd_rgba_region, z_range=(-0.1, 0.07))
            pcd_rgba_region = pcd_rgba_region[pcd_rgba_ind]
            color_c4 = np.ones((len(pcd_rgba_region), 4))
            color_c4[:, :3] = pcd_rgba_region[:, 3:]
            gm.gen_pointcloud(pcd_rgba_region[:, :3], color_c4).attach_to(base)
            # env.deep_plate.attach_to(base)
            # env.microplate.attach_to(base)
            if pcd_rgba_region is None:
                exit(0)

            print(f"pcd shape: {pcd_rgba_region.shape}")

        else:
            rack_transform = conf.rack_transform
        self.rack_transform = rack_transform

        # plot
        print(f"rack_transform: np.{repr(rack_transform)}")

        env.tip_rack.set_homomat(rack_transform)
        env.tip_rack.attach_to(base)

        self.plot_rbt_sync()
        self.show_all_tips()


    def sync(self):
        jnt = self.robot_x.get_jnt_values()
        self.robot_s.fk(jnt_values=jnt)


if __name__ == '__main__':
    from obs_rack import Rack_Locator
    import wrs._misc.promote_rt as prt

    '''
    set up    
    '''
    prt.set_realtime()
    # base = wd.World(cam_pos=(1, -0.6, 0.7), lookat_pos=[.2, 0, .1], w=1920, h=1200, )
    base = wd.World(cam_pos=(1.08, 0.6, 0.66), lookat_pos=[.38, .08, .21], w=1920, h=1200, )
    gm.gen_frame().attach_to(base)

    local_time = time.localtime()
    # save_path = f"./data/liquid_weight/{local_time[0]}_{local_time[1]}_{local_time[2]}"
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # Realsense sensor
    rs_pipeline = RealSenseD405Crop()

    component_name = "arm"
    manipulator_name = 'hnd',
    robot_s = cbtr.CobottaRIPPS()
    env = eb.Env(robot_s)
    env.update_env(microplate_pos=conf.microplate_pos, microplate_rotmat=conf.microplate_rot)
    # env.deep_plate.attach_to(base)
    env.microplate.homomat=conf.microplate_homo
    # env.deep_plate.set_homomat(conf.microplate_homo)
    # env.deep_plate_rbt.set_homomat(conf.microplate_homo)
    # env.deep_plate.attach_to(base)
    env.microplate.attach_to(base)
    print("rack pose: ", env.tip_rack.get_pos())
    print("tip1 pose: ", env.tip_rack._hole_pos_list[0])

    robot_x = cbtx.CobottaX()
    print("connected")

    long_pipette_toggle = conf.long_pipette_toggle
    test_toggle = False
    if test_toggle:
        chemical_pos = env.microplate._hole_pos_list[0 % 24]
        print(chemical_pos)
        chemical_pos[0] += 0.003
        chemical_pos[2] = 0.24

        move_to_new_pose(chemical_pos)
        time.sleep(0.1)

    balance_list = []

    mplan = MotionPlannerRT(robot_s, robot_x, obstacle_list=[env.tip_rack], init_height=0.035)
    rrtc_planner = rrtc.RRTConnect(robot_s)

    eject_jnt_values_list = conf.eject_jnt_values_list
    eject_len = len(eject_jnt_values_list)
    start_pose = conf.start_pose
    # model_resnet = model.ResnetModel(conf.model_path["resnet"])
    model_num = model.TransformerModel(model_path="./trained_model/num_model_ver2",
                                       dim=128,
                                       img_size=(200, 100),
                                       patch_size=10,
                                       num_classes=10,
                                       img_transformer=transforms.Compose(
                                           [transforms.Resize((200, 100)), transforms.ToTensor()]))

    model_vit = model.TransformerModel(conf.model_path["vit_d405"], (120, 120), 8, 61)

    exe = Execution(robot_s, robot_x, rs_pipeline, env, mplan)
    gui = HLabGUI(exe)


    def track_rbt(track_func, task):
        track_func()
        return task.again


    taskMgr.doMethodLater(.02, track_rbt, "track_rbt", extraArgs=[exe.plot_rbt_sync, ], appendTask=True)

    base.run()
