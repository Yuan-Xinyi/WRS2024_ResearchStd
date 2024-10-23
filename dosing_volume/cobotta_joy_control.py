import time

import numpy as np

import visualization.panda.world as wd
# from drivers.devices.realsense_d405.d405_driver import RealSenseD405
from robot_sim.robots.cobotta.cobotta_ripps import CobottaRIPPS
import cobotta_x_new as cbtx
import config_file as conf
from typing import Union, Callable
from panda3d.core import Filename


class Boost:
    """
    Boost the showbase
    """

    def __init__(self, base):
        self.base = base

    def screen_shot(self, img_name: str):
        """
        Take a screenshot. It can also use base.win.screenshot to save the screen shot
        :param img_name: the name of the image
        """
        self.base.graphicsEngine.renderFrame()
        self.base.win.saveScreenshot(Filename(img_name + ".jpg"))

    def add_key(self, keys: Union[str, list]):
        """
        Add key to  the keymap. The default keymap can be seen in visualization/panda/inputmanager.py
        :param keys: the keys added to the keymap
        """
        assert isinstance(keys, str) or isinstance(keys, list)

        if isinstance(keys, str):
            keys = [keys]

        def set_keys(base, k, v):
            base.inputmgr.keymap[k] = v

        for key in keys:
            if key in self.base.inputmgr.keymap: continue
            self.base.inputmgr.keymap[key] = False
            self.base.inputmgr.accept(key, set_keys, [self.base, key, True])
            self.base.inputmgr.accept(key + '-up', set_keys, [self.base, key, False])

    def add_task(self, task: Callable, args: list = None, timestep: float = 0.1):
        """
        Add a task to the taskMgr. The name of the function will be the name in the taskMgr
        :param task: a function added to the taskMgr
        :param args: the arguments of function
        :param timestep: time step in the taskMgr
        """
        if args is not None:
            self.base.taskMgr.doMethodLater(timestep, task, task.__code__.co_name,
                                            extraArgs=args,
                                            appendTask=True)
        else:
            self.base.taskMgr.doMethodLater(timestep, task, task.__code__.co_name)

    def bind_task_2_key(self, key: "str", func: Callable, args: list = None, timestep: float = .01):
        self.add_key(key)

        def bind_task(task):
            if self.base.inputmgr.keymap[key]:
                if args is None:
                    func()
                else:
                    func(*args)
            return task.again

        self.add_task(bind_task, args, timestep=timestep)


# Init base
base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0], lens_type="perspective")  # , lens_type="orthographic"

robot_x = cbtx.CobottaX()
base.boost = Boost(base)

grand = [.0005]


def move_rbt_0():
    pos = robot_x.get_pose_values()
    pos[0] += grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_0_n():
    pos = robot_x.get_pose_values()
    pos[0] -= grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_1():
    pos = robot_x.get_pose_values()
    pos[1] += grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_1_n():
    pos = robot_x.get_pose_values()
    pos[1] -= grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_max():
    grand[0] = .0005
    print(grand)


def move_rbt_min():
    grand[0] = .0001
    print(grand)


def move_rbt_2():
    pos = robot_x.get_pose_values()
    pos[2] += grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_2_n():
    pos = robot_x.get_pose_values()
    pos[2] -= grand[0]
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_5():
    pos = robot_x.get_pose_values()
    pos[5] += np.radians(5)
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


def move_rbt_5_n():
    pos = robot_x.get_pose_values()
    pos[5] -= np.radians(5)
    robot_x.move_pose(pos)
    print(repr(robot_x.get_pose_values()))


rbt_list = []


def save_rbt_pose():
    pos = robot_x.get_pose_values()
    rbt_list.append(pos)
    print(repr(rbt_list))
    move_rbt_5_n()


base.boost.bind_task_2_key("w", move_rbt_0, timestep=.1)
base.boost.bind_task_2_key("s", move_rbt_0_n, timestep=.1)
base.boost.bind_task_2_key("a", move_rbt_1, timestep=.1)
base.boost.bind_task_2_key("d", move_rbt_1_n, timestep=.1)
base.boost.bind_task_2_key("z", move_rbt_2, timestep=.1)
base.boost.bind_task_2_key("x", move_rbt_2_n, timestep=.1)
base.boost.bind_task_2_key("q", move_rbt_5, timestep=.1)
base.boost.bind_task_2_key("e", move_rbt_5_n, timestep=.1)
base.boost.bind_task_2_key("r", move_rbt_max, timestep=.1)
base.boost.bind_task_2_key("t", move_rbt_min, timestep=.1)
base.boost.bind_task_2_key("f", save_rbt_pose, timestep=.1)

base.run()
