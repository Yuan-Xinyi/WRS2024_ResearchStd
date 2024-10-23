import copy
import cv2
import numpy as np
import time
import math
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as gm
import wrs.basis.robot_math as rm
from wrs.robot_sim.robots.cobotta.cobotta import Cobotta
from wrs.robot_con.cobotta.cobotta_x import CobottaX
# import motion_planner_sim as motionrt
# import env_bulid as eb
# import fisheye_camera as camera
import struct
import os
import shutil
import matplotlib.pyplot as plt
# import concentric_circle_hex as hex
import config_file as conf
import file_sys as fs
import glob, re


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = fs.Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = fs.Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def gen_regpoly(radius, nedges=12):
    angle_list = np.linspace(0, np.pi * 2, nedges + 1, endpoint=True)
    x_vertex = np.sin(angle_list) * radius
    y_vertex = np.cos(angle_list) * radius
    return np.column_stack((x_vertex, y_vertex))


def gen_2d_isosceles_verts(nlevel, edge_length, nedges=12):
    xy_array = np.asarray([[0, 0]])
    for level in range(nlevel):
        xy_vertex = gen_regpoly(radius=edge_length * (level + 1), nedges=nedges)
        for i in range(nedges):
            xy_array = np.append(xy_array,
                                 np.linspace(xy_vertex[i, :], xy_vertex[i + 1, :], num=level + 1, endpoint=False),
                                 axis=0)
    return xy_array


def gen_2d_equilateral_verts(nlevel, edge_length):
    return gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=6)


def gen_3d_isosceles_verts(pos, rotmat, nlevel=5, edge_length=0.001, nedges=12):
    xy_array = gen_2d_isosceles_verts(nlevel=nlevel, edge_length=edge_length, nedges=nedges)
    xyz_array = np.pad(xy_array, ((0, 0), (0, 1)), mode='constant', constant_values=0)
    return rotmat.dot((xyz_array).T).T + pos


def gen_3d_equilateral_verts(pos, rotmat, nlevel=5, edge_length=0.001):
    return gen_3d_isosceles_verts(pos=pos, rotmat=rotmat, nlevel=nlevel, edge_length=edge_length, nedges=6)


def spiral(num=100):
    x = []
    y = []
    r = 0.000
    theta = 0
    for i in range(num):
        theta = theta + math.pi / (2 + 0.25 * i)
        r = r + 0.0002 / (1 + 0.1 * i)
        x.append(r * math.cos(theta))
        y.append(r * math.sin(theta))
    return x, y


def get_spiral_pose_values_list(start_pose, layer, edge_length):
    start_pos = np.asarray(start_pose[:3])
    # start_rotmat = rm.rotmat_from_euler(start_pose[3],start_pose[4],start_pose[5])
    spiral_points = gen_3d_equilateral_verts(start_pos, rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                             layer, edge_length)
    total_num = len(spiral_points)
    output_pose_list = np.asarray([start_pose] * total_num)
    output_pose_list[:, :3] = spiral_points
    return output_pose_list


def capture_spiral(spiral_id, sample_num, save_address="./capture/spiral_t/"):
    save_address = save_address + f"{spiral_id}/"
    if not os.path.exists(save_address):
        os.mkdir(save_address)
    for i in range(sample_num):
        time.sleep(0.2)
        rt_sys._pc_server_socket.send(struct.pack("!3s", b"cap"))
        while 1:
            buf = rt_sys.get_buffer()
            if buf == b"over":
                # print(i)
                break
        fcam.save_combine_row2(save_address, i)


# create_folder()
# move_sample()
# change_name("./capture/sample/")
# combine_pic("./capture/sample/",15)

# x,y=spiral(150)
# print(np.max(x),np.min(x))
# print(np.max(y),np.min(y))
# plt.plot(x,y)
# plt.show()
# # print(x)
# # print(y)
# start_pose_values = np.array([2.37048790e-01, 2.95005888e-02, 2.30548084e-01, 1.57080877e+00, -9.17807370e-07, 1.57080293e+00, 5.00000000e+00])  ## chemical
start_pose_values = [2.85649552e-01, 3.29883515e-03, 1.93549077e-01, 1.57080258e+00, -2.53598291e-07, 1.57078908e+00,
                     5.00000000e+00]  ## tip
# get_spiral_pose_values_list(start_pose_values,layer=3,edge_length=0.9)

# x_list, y_list = hex.concentric_circle_hex_polar(3, 0.9)
# spiral_list = np.zeros((len(x_list), 3))

# np.savetxt("spiral.txt", spiral_list)


if __name__ == "__main__":
    from realsensecrop import RealSenseD405Crop
    from robot_sim.robots.cobotta.cobotta_ripps import CobottaRIPPS
    import cobotta2 as cbtx
    from numpy import array

    rs_pipe = RealSenseD405Crop()
    img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
    img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
    img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
    img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
    img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
    component_name = "arm"

    rbtx = cbtx.CobottaX()
    rbt = CobottaRIPPS()

    data = fs.load_json("resources/collect_data.json")

    # obs_pose_list = [array([2.90139458e-01, 2.37984829e-02, 1.95e-01, 1.57136404e+00,
    #                         -7.59717102e-04, 1.67314437e+00, 5.00000000e+00]),
    #                  # -5
    #                  array([2.89834901e-01, 2.39617520e-02, 1.95e-01, 1.57143015e+00,
    #                         -7.62108699e-04, 1.76056769e+00, 5.00000000e+00]),
    #                  array([2.89263056e-01, 2.41587442e-02, 1.95e-01, 1.57117575e+00,
    #                         -7.89022778e-04, 1.84763023e+00, 5.00000000e+00]),
    #                  array([2.88755902e-01, 2.43286050e-02, 1.95e-01, 1.57123695e+00,
    #                         -7.96767285e-04, 1.93474514e+00, 5.00000000e+00]),
    #                  array([2.88028979e-01, 2.42246501e-02, 1.95e-01, 1.57076930e+00,
    #                         -1.34664764e-03, 2.02173598e+00, 5.00000000e+00]),
    #                  array([2.87456278e-01, 2.43696002e-02, 1.95e-01, 1.57086489e+00,
    #                         -1.50423974e-03, 2.10920691e+00, 5.00000000e+00]),
    #                  array([2.86945020e-01, 2.41082744e-02, 1.95e-01, 1.57091968e+00,
    #                         -1.93791217e-03, 2.19614215e+00, 5.00000000e+00]),
    #                  array([2.86437144e-01, 2.41210741e-02, 1.95e-01, 1.57093942e+00,
    #                         -2.02066695e-03, 2.28374175e+00, 5.00000000e+00]),
    #                  array([2.85870617e-01, 2.37973057e-02, 1.95e-01, 1.57105282e+00,
    #                         -2.24472870e-03, 2.37109510e+00, 5.00000000e+00]),
    #                  array([2.85672531e-01, 2.34797862e-02, 1.95e-01, 1.57099521e+00,
    #                         -2.31985246e-03, 2.45840017e+00, 5.00000000e+00]),
    #                  # +5
    #                  array([2.90717726e-01, 2.35750482e-02, 1.95e-01, 1.57087309e+00,
    #                         2.18890729e-04, 1.58594834e+00, 5.00000000e+00]),
    #                  array([2.91299835e-01, 2.33037504e-02, 1.95e-01, 1.57084171e+00,
    #                         1.01762751e-04, 1.49852843e+00, 5.00000000e+00]),
    #                  array([2.91606657e-01, 2.31958329e-02, 1.95e-01, 1.57059025e+00,
    #                         1.27251801e-04, 1.41137271e+00, 5.00000000e+00]),
    #                  array([2.92103353e-01, 2.28664332e-02, 1.95e-01, 1.57049897e+00,
    #                         4.59392359e-05, 1.32388184e+00, 5.00000000e+00]),
    #                  array([2.92578470e-01, 2.24983986e-02, 1.95e-01, 1.57043430e+00,
    #                         -2.73394029e-04, 1.23649646e+00, 5.00000000e+00]),
    #                  array([2.92761150e-01, 2.22118137e-02, 1.95e-01, 1.57035458e+00,
    #                         -2.71966999e-04, 1.14907525e+00, 5.00000000e+00]),
    #                  array([2.92930422e-01, 2.18331612e-02, 1.95e-01, 1.57016125e+00,
    #                         -1.40657843e-04, 1.06145304e+00, 5.00000000e+00]),
    #                  array([2.93617034e-01, 2.14471607e-02, 1.95e-01, 1.56993458e+00,
    #                         -1.38767921e-04, 9.74080372e-01, 5.00000000e+00]),
    #                  array([2.93871890e-01, 2.09028616e-02, 1.95e-01, 1.57005930e+00,
    #                         -2.85474522e-04, 8.86570722e-01, 5.00000000e+00])
    #                  ]
    obs_pose_list = []
    for k, v in data.items():
        if 'data_' in k:
            obs_pose_list.append([np.asarray(v['pos']), np.asarray(v['rot'])])

    init_pose = copy.deepcopy(obs_pose_list[0])
    init_pose[0][2] = conf.record_height
    rbtx.move_p(init_pose[0], init_pose[1])

    path = fs.Path('./data/')
    save_path = increment_path(path.joinpath('capture'), exist_ok=True, mkdir=True)
    feature_path = save_path.joinpath("feature")
    feature_big_path = save_path.joinpath("feature_big")
    img_1_path = save_path.joinpath("img_1")
    [p.mkdir(exist_ok=True) for p in [feature_path, feature_big_path, img_1_path]]
    pic_id = 171
    while 1:
        input(f"{pic_id} capture start!!!!!!!!!!!")
        for obs_pose in obs_pose_list:
            start_pose = copy.deepcopy(obs_pose)
            rbtx.move_p(start_pose[0], start_pose[1])
            time.sleep(.1)
            save_address_f = feature_path.joinpath(str(pic_id))
            save_address_fb = feature_big_path.joinpath(str(pic_id))
            save_address_i1 = img_1_path.joinpath(str(pic_id))
            save_address_f.mkdir(exist_ok=True)
            save_address_fb.mkdir(exist_ok=True)
            save_address_i1.mkdir(exist_ok=True)

            sample_pos_list = get_spiral_pose_values_list(
                rbtx.wrshomomat2cobbotapos(rm.homomat_from_posrot(*start_pose)), layer=4, edge_length=0.0009)
            for id, pose in enumerate(sample_pos_list):
                # print(pose)
                pose = copy.deepcopy(pose)
                pose[2] = conf.record_height
                rbtx.move_pose(pose)
                img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
                img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
                img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
                img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
                img_f, img_fb, img_1 = rs_pipe.get_learning_feature()
                cv2.imwrite(str(save_address_f.joinpath(f'{id}.jpg')), img_f)
                cv2.imwrite(str(save_address_fb.joinpath(f'{id}.jpg')), img_fb)
                cv2.imwrite(str(save_address_i1.joinpath(f'{id}.jpg')), img_1)
            pic_id += 1
        rbtx.move_p(init_pose[0], init_pose[1])
        print(f"{pic_id - 1} capture finish!!!!!!!!!!!")
        print("-" * 30)
