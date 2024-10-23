import time

import numpy as np
from direct.task.TaskTester import counter

from wrs import rm, wd, mip
import wrs.robot_con.cobotta.cobotta_x as cbtx
import wrs.robot_sim.robots.cobotta.cobotta_wrsgripper_v4 as cbtw
import Largehand.largehand_gripper_x as grpx
from wrs.drivers.devices.dynamixel_sdk.sdk_wrapper import DynamixelMotor, PortHandler

import confog_file_gripper as conf


def update(obj_cnter, onscreen_model, anime_data_list, counter, task):
    if obj_cnter[0] >= len(anime_data_list):
        obj_cnter[0] = 0
    anime_data = anime_data_list[obj_cnter[0]]
    # if base.inputmgr.keymap["space"] is True:
    time.sleep(.1)
    for model in onscreen_model:
        model.detach()
    robot_s.fk(jnt_values=anime_data)
    cbt_model = robot_s.gen_meshmodel()
    onscreen_model.append(cbt_model)
    cbt_model.attach_to(base)
    counter[0] += 1
    if counter[0] >= len(anime_data_list):
        # anime_data.support_facets[anime_data.counter - 1].detach()
        counter[0] = 0
        obj_cnter[0] += 1
    return task.cont


def cbt_pose_simulation(pose_list):
    jnt_values_list = []
    for pose in pose_list:
        jnt_values = robot_x.P2J(pose)
        jnt_values_list.append(jnt_values)

    onscreen_model = []

    obj_cnter = [0]
    counter = [0]
    taskMgr.doMethodLater(0.5, update, "update",
                          extraArgs=[obj_cnter, onscreen_model, jnt_values_list, counter],
                          appendTask=True)
    base.run()


if __name__ == "__main__":
    base = wd.World(cam_pos=(1.08, 0.6, 0.66), lookat_pos=[.38, .08, .21], w=1920, h=1200, )

    gl_speed = 50
    device = 'COM4'
    baudrate = 57600
    port_handler = PortHandler(device)
    gripper_id = 10
    gripper_calib_data_path = "resources/dynamixel_calib/gripper_x_calib_data"
    gripper_x = grpx.LargehandGripperX(device=device,
                                       baudrate=baudrate,
                                       port_handler=port_handler,
                                       motor_ids=[gripper_id],
                                       gripper_limit=(0, 0.134),
                                       calib_data_path=gripper_calib_data_path)
    if not gripper_x.is_calibrated:
        gripper_x.calibrate(close_gripper_direction=1)
    robot_x = cbtx.CobottaX(host="192.168.0.11")
    robot_s = cbtw.CobottaLarge()
    mip_planner = mip.InterplatedMotion(robot_s)

    #
    robot_x.move_pose(conf.init_pose)
    gripper_x.close_gripper()

    # simulation check
    # cbt_pose_simulation(conf.open_door_list)

    for pose in conf.open_door_list:
        robot_x.move_pose(pose,speed=80)

    # move to microplate storage area
    robot_x.move_pose(conf.microplate_pose_temp,speed=gl_speed)
    gripper_x.open_gripper()
    robot_x.move_pose(conf.microplate_pose,speed=gl_speed)
    gripper_x.close_gripper()
    robot_x.move_pose(conf.microplate_pose_temp,speed=30)

    robot_x.move_pose(conf.storage_pose_temp,speed=gl_speed)
    robot_x.move_pose(conf.storage_pose,speed=gl_speed)
    gripper_x.open_gripper()
    robot_x.move_pose(conf.storage_pose_temp,speed=gl_speed)
    gripper_x.set_gripper_width(0.03)

    robot_x.move_pose(conf.push_start_pose_temp,speed=gl_speed)
    # gripper_x.set_gripper_width(0.03)
    robot_x.move_pose(conf.push_start_pose,speed=gl_speed)

    robot_x.move_pose(conf.push_end_pose,speed=15)
    robot_x.move_pose(conf.push_start_pose_temp,speed=gl_speed)
    #
    #
    gripper_x.open_gripper()
    robot_x.move_pose(conf.regrasp_pose,speed=gl_speed)
    gripper_x.close_gripper()

    robot_x.move_pose(conf.regrasp_lift_pose,speed=gl_speed)
    robot_x.move_pose(conf.balance_pose_temp,speed=gl_speed)

    ## linear path planning
    # regrasp_lift_jnt_values = robot_x.P2J(conf.regrasp_lift_pose)
    # balance_jnt_values = robot_x.P2J(conf.balance_pose)
    # balance_placement_result = mip_planner.gen_interplated_between_given_conf(start_jnt_values=regrasp_lift_jnt_values,
    #                                            end_jnt_values=balance_jnt_values,
    #                                            )
    # balance_placement_path=balance_placement_result._jv_list
    # print(balance_placement_path)
    # for balance_jnt in balance_placement_path:
    #     robot_s.fk(jnt_values=balance_jnt)
    #     robot_s.gen_meshmodel().attach_to(base)
    # base.run()
    # robot_x.move_jnts_motion(balance_placement_path)


    robot_x.move_pose(conf.balance_pose,speed=30)
    robot_x.move_pose(conf.balance_pose_place,speed=10)
    gripper_x.set_gripper_width(0.11)
    time.sleep(0.5)
    # breakpoint()
    robot_x.move_pose(conf.balance_pose_finish, speed=30)
    gripper_x.close_gripper()
    robot_x.move_pose(conf.balance_pose_temp,speed=gl_speed)
    robot_x.move_pose(conf.regrasp_lift_pose)

    gripper_x.close_gripper()
    for close_pose in conf.close_door_list:
        robot_x.move_pose(close_pose,speed=50)
