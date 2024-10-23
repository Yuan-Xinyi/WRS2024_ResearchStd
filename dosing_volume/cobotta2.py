import copy
import time
import math
import numpy as np
from wrs import rm
import wrs.motion.trajectory.topp_ra as trajp
import wrs.drivers.orin_bcap.bcapclient as bcapclient
import numpy.typing as npt
from typing import List
import config_file as conf


class CobottaX(object):

    def __init__(self, host='192.168.0.1', port=5007, timeout=2000):
        """
        :param host:
        :param port:
        :param timeout:

        author: weiwei
        date: 20210507
        """
        self.bcc = bcapclient.BCAPClient(host, port, timeout)
        self.bcc.service_start("")
        # Connect to RC8 (RC8(VRC)provider)
        self.hctrl = self.bcc.controller_connect("", "CaoProv.DENSO.VRC", ("localhost"), (""))
        self.clear_error()
        # get robot_s object hanlde
        self.hrbt = self.bcc.controller_getrobot(self.hctrl, "Arm", "")
        # print(self.bcc.robot_getvariablenames(self.hrbt))
        # self.bcc.controller_getextension(self.hctrl, "Hand", "")
        # take arm
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        # motor on
        self.bcc.robot_execute(self.hrbt, "Motor", [1, 0])
        # set ExtSpeed = [speed, acc, dec]
        self.bcc.robot_execute(self.hrbt, "ExtSpeed", [100, 100, 100])
        # self.traj_gen = trajp.PiecewisePolyTOPPRA()
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,1,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,2,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,3,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,4,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,5,0.01])
        # self.bcc.robot_execute(self.hrbt,"ErAlw",[True,6,0.01])

    @staticmethod
    def wrshomomat2cobbotapos(wrshomomat, k=-1):
        pos = wrshomomat[:3, 3]
        rpy = rm.rotmat_to_euler(wrshomomat[:3, :3])
        return np.hstack([pos, rpy, k])

    @staticmethod
    def cobottapos2wrspos(cobbota_pos):
        pos = cobbota_pos[:3]
        rpy = cobbota_pos[3:6]
        return pos, rm.rotmat_from_euler(*rpy)

    def __del__(self):
        self.clear_error()
        self.bcc.controller_getrobot(self.hrbt, "Motor", [0, 0])
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        self.bcc.robot_release(self.hrbt)
        self.bcc.controller_disconnect(self.hctrl)
        self.bcc.service_stop()

    def clear_error(self):
        self.bcc.controller_execute(self.hctrl, "ClearError", None)

    def disconnect(self):
        self.bcc.controller_disconnect(self.hctrl)

    def moveto_named_pose(self, name):
        self.bcc.robot_move(self.hrbt, 1, name, "")

    def move_jnts_motion(self, path: List[npt.NDArray[float]], toggle_debug: bool = False):
        """
        :param path:
        :return:
        author: weiwei
        date: 20210507
        """
        time.sleep(0.1)
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])  # 20220317, needs further check, speedmode?
        time.sleep(0.2)
        new_path = []
        for i, pose in enumerate(path):
            if i < len(path) - 1 and not np.allclose(pose, path[i + 1]):
                new_path.append(pose)
        new_path.append(path[-1])
        path = new_path
        # max_vels = [math.pi * .6, math.pi * .4, math.pi, math.pi, math.pi, math.pi * 1.5]
        max_vels = [math.pi / 6, math.pi / 6, math.pi / 6, math.pi / 3, math.pi / 3, math.pi / 2]
        _,interpolated_confs,_,_ = \
            trajp.generate_time_optimal_trajectory(path,
                                                   max_vels=max_vels,
                                                   ctrl_freq=.008)
        # print(f"toppra{interpolated_confs[:,2].max()}")
        # Slave move: Change mode
        while True:
            try:
                # time.sleep(.2)
                self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x202)
                time.sleep(.5)
                # print("sleep done")
                print(self.get_jnt_values())
                print(interpolated_confs[0].tolist())
                self.bcc.robot_execute(self.hrbt, "slvMove", np.degrees(interpolated_confs[0]).tolist() + [0, 0])
                # time.sleep(.2)
                # print("try exec done")
                break
            except:
                print("exception, continue")
                self.clear_error()
                time.sleep(0.2)
                continue
        try:
            for jnt_values in interpolated_confs:
                jnt_values_degree = np.degrees(jnt_values)
                self.bcc.robot_execute(self.hrbt, "slvMove", jnt_values_degree.tolist() + [0, 0])
            # print("trajectory done")
        except:
            # print("trajectory exception, continue")
            # self.clear_error()
            time.sleep(0.2)
            return False
        self.bcc.robot_execute(self.hrbt, "slvChangeMode", 0x000)
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)
        return True

    def get_jnt_values(self):
        pose = self.bcc.robot_execute(self.hrbt, "CurJnt", None)
        return np.radians(np.array(pose[:6]))

    def get_pose(self):
        """
        x,y,z,r,p,y,fig
        :return:
        author: weiwei
        date: 20220115
        """
        return self.cobottapos2wrspos(self.get_pose_values())

    def get_pose_values(self):
        """
        x,y,z,r,p,y,fig
        :return:
        author: weiwei
        date: 20220115
        """
        pose = self.bcc.robot_execute(self.hrbt, "CurPos", None)
        return_value = np.array(pose[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def move_jnts(self, jnt_values: npt.NDArray[float]):
        """
        :param jnt_values:  1x6 np array
        :return:
        author: weiwei
        date: 20210507
        """
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        jnt_values_degree = np.degrees(jnt_values)
        self.bcc.robot_move(self.hrbt, 1, [jnt_values_degree.tolist(), "J", "@E"], "")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def move_p(self, pos, rot, speed=100):
        pose = self.wrshomomat2cobbotapos(rm.homomat_from_posrot(pos, rot))
        self.move_pose(pose, speed)

    def move_pose(self, pose, speed=100):
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.1)
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        self.bcc.robot_move(self.hrbt, 1, [pose_value.tolist(), "P", "@E"], f"SPEED={speed}")
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)

    def open_gripper(self, dist=.03, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def close_gripper(self, dist=.0, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def defult_gripper(self, dist=.012, speed=100):
        """
        :param dist:
        :return:
        """
        assert 0 <= dist <= .03
        self.bcc.controller_execute(self.hctrl, "HandMoveA", [dist * 1000, speed])

    def P2J(self, pose):
        pose = np.array(pose)
        pose_value = copy.deepcopy(pose)
        pose_value[:3] *= 1000
        pose_value[3:6] = np.degrees(pose_value[3:6])
        return np.radians(self.bcc.robot_execute(self.hrbt, "P2J", pose_value.tolist()))[:6]

    def ik(self, pos, rot):
        pose = self.wrshomomat2cobbotapos(rm.homomat_from_posrot(pos, rot))
        self.P2J(pose)

    def J2P(self, jnt_values):
        jnt_values = np.array(jnt_values)
        jnt_values_degree = np.degrees(jnt_values)
        pose_value = np.radians(self.bcc.robot_execute(self.hrbt, "J2P", jnt_values_degree.tolist()))
        return_value = np.array(pose_value[:7])
        return_value[:3] *= .001
        return_value[3:6] = np.radians(return_value[3:6])
        return return_value

    def null_space_search(self, current_pose):
        pose = copy.deepcopy(current_pose)
        times = 0
        for angle in range(0, 180, 5):
            for i in [-1, 1]:
                try:
                    self.P2J(pose)
                    return pose, times
                except:
                    self.clear_error()
                    times += 1
                    time.sleep(0.1)
                    pose[5] = current_pose[5] + np.radians(angle * i)
        return None, times

    def move_p_nullspace(self, tgt_pos, tgt_rot, k=-1, speed=100):
        pose = self.wrshomomat2cobbotapos(wrshomomat=rm.homomat_from_posrot(tgt_pos, tgt_rot), k=k)
        self.move_to_pose_nullspace(pose, speed)

    def move_to_pose_nullspace(self, pose, speed=100):
        pose, times = self.null_space_search(pose)
        if pose is not None:
            self.move_pose(pose, speed=speed)
            return times
        else:
            raise Exception("No solution!")

    def vertical_pose(self):
        pose = self.get_pose_values()
        pose[4] = 0
        pose[3] = np.pi / 2
        self.move_pose(pose)

    def is_pose_reachable(self, pose):
        pose_value = copy.deepcopy(pose)
        pose_value = np.array(pose_value)
        if pose_value.shape[0] == 7:
            pose_value[:3] *= 1000
            pose_value[3:6] = np.degrees(pose_value[3:6])
            out_range = self.bcc.robot_execute(self.hrbt, "OutRange", f"P{tuple(pose_value)}")
        elif pose_value.shape[0] == 6:
            jnt_values_degree = np.degrees(pose_value)
            out_range = self.bcc.robot_execute(self.hrbt, "OutRange", f"J{tuple(jnt_values_degree)}")
        else:
            print("The position data is wrong")
            return False
        if out_range == 0:
            print("reachable")
            return True
        else:
            print(f"Axis {out_range} out of range")
            return False

    def move_pose_list(self, pose_list, speed=100):
        self.hhnd = self.bcc.robot_execute(self.hrbt, "TakeArm", [0, 0])
        time.sleep(0.3)
        for id, pose in enumerate(pose_list):
            print(id, pose)
            pose = np.array(pose)
            pose_value = copy.deepcopy(pose)
            pose_value[:3] *= 1000
            pose_value[3:6] = np.degrees(pose_value[3:6])
            while True:
                try:
                    self.bcc.robot_move(self.hrbt, 1, [pose_value.tolist(), "P", "@E"], f"SPEED={speed}")
                    break
                except:
                    self.clear_error()
        self.bcc.robot_execute(self.hrbt, "GiveArm", None)
        time.sleep(0.1)


def open_door(robot_x, speed=50, debug_toggle=False):
    robot_x.move_pose(conf.execute_pose[0])
    if debug_toggle:
        for id in np.arange(conf.open_id, conf.close_id, dtype=int):
            robot_x.move_pose(conf.execute_pose[id], speed)
            print(f"position{id} pose: np.{repr(robot_x.get_pose_values())}")
            print(f"position{id} jnt_values: np.{repr(robot_x.get_jnt_values())}")
            input()
    else:
        robot_x.move_pose_list(conf.execute_pose[conf.open_id:conf.close_id], speed)


def close_door(robot_x, speed=50, debug_toggle=False):
    robot_x.move_pose(conf.execute_pose[0])
    if debug_toggle:
        for id in np.arange(conf.close_id, conf.balance_id, dtype=int):
            robot_x.move_pose(conf.execute_pose[id], speed)
            print(f"position{id} pose: np.{repr(robot_x.get_pose_values())}")
            print(f"position{id} jnt_values: np.{repr(robot_x.get_jnt_values())}")
            input()
    else:
        robot_x.move_pose_list(conf.execute_pose[conf.close_id:conf.balance_temp_id], speed)


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


if __name__ == '__main__':
    import math
    import numpy as np
    from wrs import rm, mgm, wd
    import wrs.robot_sim.robots.cobotta.cobotta_ripps as cbt
    import wrs.motion.probabilistic.rrt_connect as rrtc


    base = wd.World(cam_pos=[1, 1, .5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)

    # connect to robot
    robot_s = cbt.CobottaRIPPS()
    robot_x = CobottaX(host="192.168.0.11")  # robot_x = CobottaX(host="192.168.0.11")

    # # current pose info
    np.set_printoptions(linewidth=np.inf)
    print(f"current pose: np.{repr(robot_x.get_pose_values())}")
    print(f"current jnt_values: np.{repr(robot_x.get_jnt_values())}")
    print(repr(np.degrees(robot_x.get_jnt_values())))
    current_pose = robot_x.get_pose_values()
    robot_x.is_pose_reachable(current_pose)

    # # gripper test (xinyi: test the connection)
    # robot_x.open_gripper(speed=80)
    # robot_x.close_gripper(speed=100)
    # robot_x.defult_gripper(speed=80)

    # open the door test
    # open_door(robot_x,debug_toggle=False)

    # close the door
    # close_door(robot_x, debug_toggle=False)

    # # front panal motion test
    # robot_x.move_jnts(conf.photo_start_jnts_values)
    # robot_x.move_pose(conf.photo_temp_pose_values)
    # robot_x.move_pose(conf.photo_end_pose_values)
