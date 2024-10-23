"""
functions to enhance functionality for the Panda3d ShowBase.
"""
from typing import Union, Callable

from panda3d.core import (Filename, NodePath, WindowProperties, Point3, Vec3,
                          CardMaker, Texture, PNMImage, Point2, PTAUchar, )
import numpy as np


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

    def add_task(self, task: Callable, args: list = None, timestep: None or float = 0.1):
        """
        Add a task to the taskMgr. The name of the function will be the name in the taskMgr
        :param task: a function added to the taskMgr
        :param args: the arguments of function
        :param timestep: time step in the taskMgr
        """
        if timestep is None:
            add_task_method = self.base.taskMgr.add
        else:
            add_task_method = self.base.taskMgr.doMethodLater

        if args is not None:
            if timestep is None:
                add_task_method(task, task.__code__.co_name,
                                extraArgs=args,
                                appendTask=True)
            else:
                add_task_method(timestep, task, task.__code__.co_name,
                                extraArgs=args,
                                appendTask=True)
        else:
            if timestep is None:
                add_task_method(task, task.__code__.co_name)
            else:
                add_task_method(timestep, task, task.__code__.co_name)

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


def projection(a: np.ndarray, b: np.ndarray):
    """
    Get the vector projection of vector b on vector a
    :param a: vector a
    :param b: vector b
    :return:
    """
    return b - a * np.dot(b, a) / np.linalg.norm(a)


def zoombase(base, direction: np.ndarray, task=None):
    """
    This is a task added to the taskMgr.
    Adjust the view of camera in the panda3d to fit the objects in the scene
    * When the objects in the scene are very scattered, this function may work poorly
    :param base: ShowBase
    :param direction: the looking direction of the camera
    :param task: a arg for taskMgr
    :return: task.again (run this function again)
    """
    bounds = base.render.getTightBounds()
    if bounds is not None:
        center = (bounds[0] + bounds[1]) / 2
        # print(center)
        point1 = np.array([bounds[0][0], bounds[0][1], bounds[0][2]])
        point2 = np.array([bounds[1][0], bounds[1][1], bounds[1][2]])
        point1_project = projection(direction, point1)
        point2_project = projection(direction, point2)
        line_project = point2_project - point1_project
        length = np.linalg.norm(line_project) / 2
        # print(length)
        # horizontal_vector = np.dot(R_matrix,horizontal_vector_origin)
        # vertical__vector = np.dot(R_matrix, vertical_vector_origin)

        # project to the plane
        Fov = base.cam.node().getLens().getFov()
        horizontalAngle, verticalAngle = Fov[0], Fov[1]
        d_horzontal = length / np.tan(np.deg2rad(horizontalAngle) / 2)
        d_vertical = length / np.tan(np.deg2rad(verticalAngle) / 2)
        # angle = [horizontalAngle, verticalAngle][np.argmax([d_horzontal,d_vertical])]
        distance = max(d_horzontal, d_vertical)
        # print(distance)

        campos = np.array([center[0], center[1], center[2]]) + direction * distance
        base.cam.setPos(campos[0], campos[1], campos[2])
        base.cam.lookAt(center[0], center[1], center[2])
        print(campos[0], campos[1], campos[2])
        print(center[0], center[1], center[2])
    if task is None:
        return None
    else:
        return task.done


def zoombase2(base, direction: np.ndarray, task):
    """
    same as zoombase. It will be deprecated in the future
    """
    bounds = base.render.getTightBounds()
    if bounds is not None:
        center = (bounds[0] + bounds[1]) / 2
        # print(center)
        point1 = np.array([bounds[0][0], bounds[0][1], bounds[0][2]])
        point2 = np.array([bounds[1][0], bounds[1][1], bounds[1][2]])
        point1_project = projection(direction, point1)
        point2_project = projection(direction, point2)
        line_project = point2_project - point1_project
        length = np.linalg.norm(line_project) / 2
        # print(length)
        # horizontal_vector = np.dot(R_matrix,horizontal_vector_origin)
        # vertical__vector = np.dot(R_matrix, vertical_vector_origin)

        # project to the plane
        Fov = base.cam3.node().getLens().getFov()
        horizontalAngle, verticalAngle = Fov[0], Fov[1]
        d_horzontal = length / np.tan(np.deg2rad(horizontalAngle) / 2)
        d_vertical = length / np.tan(np.deg2rad(verticalAngle) / 2)
        # angle = [horizontalAngle, verticalAngle][np.argmax([d_horzontal,d_vertical])]
        distance = max(d_horzontal, d_vertical)
        # print(distance)

        campos = np.array([center[0], center[1], center[2]]) + direction * distance
        base.cam3.setPos(campos[0], campos[1], campos[2])
        base.cam3.lookAt(center[0], center[1], center[2])
    return task.again


def spawn_window(base, win_size=(1024, 768), cam_pos=(2, 0, 1.5), lookat_pos=(0, 0, .2), name="render2"):
    """
    Spawn a new window
    :param base: Showbase
    :param win_size: the size of the new window
    :param cam_pos: the camera pos of the new window
    :param lookat_pos: the looking direction of the camera in new window
    :param name: the name of the new window
    :return: render scene of new window, camera in new window, new windows
    """
    render2 = NodePath(name)

    window2 = base.openWindow(props=WindowProperties(base.win.getProperties()),
                              scene=render2)
    w, h = win_size
    window_props = WindowProperties()
    window_props.setSize(w, h)
    window2.requestProperties(window_props)
    cam = base.makeCamera(window2, scene=render2)
    cam.reparentTo(render2)
    cam.setPos(Point3(cam_pos[0], cam_pos[1], cam_pos[2]))
    up = np.array([0, 0, 1])
    cam.lookAt(Point3(lookat_pos[0], lookat_pos[1], lookat_pos[2]), Vec3(up[0], up[1], up[2]))
    cam.node().setLens(base.cam.node().getLens())
    window2.setClearColorActive(True)
    window2.setClearColor((1, 1, 1, 1))
    return render2, cam, window2


def boost_base(base):
    """
    boost showbase and add spawn_window as a member function to showbase
    :param base: ShowBase
    :return: Showbase
    """
    base.boost = Boost(base)
    base.spawn_window = spawn_window
    return base


def gen_img_texture_on_render(render, img_shape=(2064, 1544)):
    """
    Add a in scene image to the window. Work with the function `set_image_texture`
    * This function will be deprecated in the future. Please use `render_img` to replace
    :param render: render scene of the base
    :param img_shape: shape of image of the texture
    :return: the texture buffer to refresh new image
    """
    x_size, y_size = img_shape[0], img_shape[1]
    xy_ratio = float(y_size) / float(x_size)
    input_img = PNMImage(x_size, y_size, 4)
    input_tex = Texture()
    input_tex.load(input_img)

    card = CardMaker('in_scene_screen')
    card.setFrameFullscreenQuad()
    card.setUvRange(Point2(0, 1),  # ll
                    Point2(1, 1),  # lr
                    Point2(1, 0),  # ur
                    Point2(0, 0))  # ul
    screen = render.attach_new_node(card.generate())
    screen.set_scale(1, 1, xy_ratio)
    screen.set_pos(0, 2, 0)
    screen.setTexture(input_tex)
    return input_tex


def set_img_texture(img: np.ndarray, texture):
    """
    Set the image for the texture buffer. Work with the function `gen_img_texture_on_render`
    :param img: the image in opencv BGR channel
    :param texture: the texture buffer generated by `gen_img_texture_on_render`
    """
    texture.set_ram_image_as(PTAUchar(img.copy()),
                             'BGR')


# def render_img(showbase, img: np.ndarray) -> ImgOnscreen:
#     """
#     Add a in scene image to the window.
#     :param showbase: Showbase
#     :param img: the image in opencv BGR channel
#     :return: ImgOnScreen (use update_img to refresh the image on the screen)
#     """
#     x_size, y_size, channel = img.shape
#     xy_ratio = float(y_size) / float(x_size)
#     screen_img = ImgOnscreen(size=(x_size, y_size),
#                              parent_np=showbase.render2d)
#     screen_img.update_img(img)
#     return screen_img
