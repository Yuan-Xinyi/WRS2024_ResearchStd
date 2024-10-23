import copy
import cv2
import time
import numpy as np
import math
import os
import cobotta_x_new as cbtx
from drivers.devices.realsense.realsense_d400s import RealSenseD405
import basis.robot_math as rm

obs_height = 0.201
obs_pose_list = [np.array([ 2.8031e-01,  3.0601e-02,  2.3300e-01,  1.5708e+00, 0,  1.5708e+00,  5]),
                 # -5
                 np.array([2.8057e-01, 3.0621e-02, 2.3298e-01, 1.5706e+00, 0, 1.4837e+00, 5]),
                 np.array([2.8086e-01, 3.0227e-02, 2.3298e-01, 1.5705e+00, 0, 1.3962e+00, 5]),
                 np.array([2.8115e-01, 3.0099e-02, 2.3298e-01, 1.5705e+00, 0, 1.3091e+00, 5]),
                 np.array([2.8132e-01, 2.9767e-02, 2.3299e-01, 1.5705e+00, 0, 1.2220e+00, 5]),
                 np.array([2.8161e-01, 2.9390e-02, 2.3299e-01, 1.5705e+00, 0, 1.1347e+00, 5]),
                 np.array([2.8159e-01, 2.8688e-02, 2.3297e-01, 1.5707e+00, 0, 1.0474e+00, 5]),
                 np.array([2.8168e-01, 2.8434e-02, 2.3296e-01, 1.5708e+00, 0, 9.6013e-01, 5]),
                 np.array([2.8186e-01, 2.8127e-02, 2.3296e-01, 1.5708e+00, 0, 8.7280e-01, 5]),
                 np.array([2.8214e-01, 2.7833e-02, 2.3293e-01, 1.5708e+00,0, 7.8553e-01, 5]),

                 # +5
                 np.array([2.8012e-01, 3.0733e-02, 2.3303e-01, 1.5706e+00, 0, 1.6579e+00, 5]),
                 np.array([2.7984e-01, 3.0974e-02, 2.3306e-01, 1.5704e+00, 0, 1.7451e+00, 5]),
                 np.array([2.7945e-01, 3.1136e-02, 2.3307e-01, 1.5703e+00, 0, 1.8324e+00, 5]),
                 np.array([2.7907e-01, 3.1277e-02, 2.3309e-01, 1.5702e+00, 0, 1.9196e+00, 5]),
                 np.array([2.7855e-01, 3.1463e-02, 2.3303e-01, 1.5704e+00, 0, 2.0070e+00, 5]),
                 np.array([2.7816e-01, 3.1352e-02, 2.3302e-01, 1.5705e+00, 0, 2.0944e+00, 5]),
                 np.array([2.7777e-01, 3.1371e-02, 2.3301e-01, 1.5705e+00, 0, 2.1815e+00, 5]),
                 np.array([2.7738e-01, 3.1299e-02, 2.3301e-01, 1.5705e+00, 0, 2.2689e+00, 5]),
                 np.array([2.7709e-01, 3.1351e-02, 2.3303e-01, 1.5706e+00, 0, 2.3564e+00, 5]),
                 ]

def capture_save(rs_pipe, save_path,pic_id):
    img = rs_pipe.get_color_img()



def get_spiral_pose_values_list(start_pose, layer, edge_length):
    start_pos = np.asarray(start_pose[:3])
    # start_rotmat = rm.rotmat_from_euler(start_pose[3],start_pose[4],start_pose[5])
    spiral_points = rm.gen_3d_equilateral_verts(start_pos, rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                                                layer, edge_length)
    total_num = len(spiral_points)
    output_pose_list = np.asarray([start_pose] * total_num)
    output_pose_list[:, :3] = spiral_points
    return output_pose_list


def create_dataset_spiral(robot_x, rs_pipe, obs_pose_list, save_path_parent, dataset_id):
    init_pose = copy.deepcopy(obs_pose_list[0])
    init_pose[2] = obs_height
    robot_x.move_pose(init_pose)
    for i in range(1000):
        # real_id = 50
        input(f"{dataset_id} capture start!!!!!!!!!!!")
        for obs_pose in obs_pose_list:
            start_pose = copy.deepcopy(obs_pose)
            start_pose[2] = 0.236
            robot_x.move_pose(start_pose)
            time.sleep(3)
            save_address = save_path_parent + f"{dataset_id}/"
            save_address_L = save_path_parent + f"{dataset_id}/"

            if not os.path.exists(save_address):
                os.mkdir(save_address)
            if not os.path.exists(save_address_L):
                os.mkdir(save_address_L)
            # input(f"frame{pic_id}")
            sample_pos_list = get_spiral_pose_values_list(start_pose, layer=4, edge_length=0.0009)
            for id, pose in enumerate(sample_pos_list):
                capture_save(rs_pipe,save_address,id)
                # # print(pose)
                # robot_x.move_pose(pose)
                # img_tip = fcam.get_frame_cut_combine_row()
                # time.sleep(0.5)
                # # fcam.save_combine_real(save_address, id)
                # fcam.save_combine_row_L(save_address, id)
                # fcam.save_combine_original(save_address_L, id)
            dataset_id += 1

        robot_x.move_pose(init_pose)
        print(f"{dataset_id - 1} capture finish!!!!!!!!!!!")
        print("-" * 30)




if __name__ == "__main__":
    robot_x = cbtx.CobottaX()
    rs_pipe = RealSenseD405()

    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    img = rs_pipe.get_color_img()
    time.sleep(1)
    np.set_printoptions(precision=4,linewidth=np.inf)

    print(rm.rotmat_to_euler(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T))
    # create_dataset(robot_x, obs_pose_list)
    # for id,obs_pose in enumerate(obs_pose_list):
    #     print(f"np.{repr(obs_pose)}",end=",\n")
    #     start_pose = copy.deepcopy(obs_pose)
    #     start_pose[2] = obs_height
    #     robot_x.move_pose(start_pose)
    #     input(f"pose{id}")
