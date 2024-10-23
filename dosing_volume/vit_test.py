if __name__ == '__main__':
    import math
    import time

    import cv2
    import numpy as np
    import torch

    import basis.robot_math as rm
    import cobotta2 as cbtx
    import model_loader as model
    import modeling.geometric_model as gm
    import robot_sim.robots.cobotta.cobotta_ripps as cbtr
    import visualization.panda.world as wd
    import config_file as conf
    from realsensecrop import RealSenseD405Crop, letterbox
    from sample import gen_2d_isosceles_verts

    '''
    set up    
    '''
    base = wd.World(cam_pos=[0.2445, 0.07175, 0.67275], lookat_pos=[0.2445, 0.07175, 0])
    gm.gen_frame().attach_to(base)

    # Realsense sensor
    rs_pipeline = RealSenseD405Crop()

    component_name = "arm"
    manipulator_name = 'hnd',
    rbts = cbtr.CobottaRIPPS()
    # rbtx = cbtx.CobottaX()
    # rbtx.defult_gripper()

    # rbtx.move_p(pos=np.array([0.29827158,
    #                           0.02247648,
    #                           0.165]), rot=np.array([[
    #     0.0,
    #     0.0,
    #     1.0
    # ],
    #     [
    #         1.0,
    #         0.0,
    #         0.0
    #     ],
    #     [
    #         0.0,
    #         1.0,
    #         0.0
    #     ]]))

    model_vit = model.TransformerModel("trained_model/vit_120_20230601-223446_best", (120, 120), 8, 61)

    total_err = np.zeros(2)
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
    while 1:
        while 1:
            # time.sleep(2)
            current_pose = rbtx.get_pose_values()
            time.sleep(0.3)
            pic = rs_pipeline.get_learning_feature()[1]
            pic = rs_pipeline.get_learning_feature()[1]

            time.sleep(0.1)
            # pic_masked = cv2.bitwise_and(pic, pic, mask=mask)
            with torch.no_grad():
                direct_trans_tip = model_vit.get_score(pic)
            score = direct_trans_tip
            print("score", score)
            pic_resized = letterbox(pic, new_shape=[360, 640], auto=False)[0]
            cv2.imshow("judge", pic_resized)
            # cv2.imshow("masked",pic_masked)
            cv2.waitKey(1)

            if score == 0:
                pos, rot = rbtx.get_pose()
                pos_r = pos.copy()
                pos_r[2] = .157
                rbtx.move_p_nullspace(pos_r, rot)
                rbtx.move_p_nullspace(pos, rot)
                rbtx.open_gripper()
                rbtx.close_gripper()
                rbtx.defult_gripper()
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
                times = rbtx.move_to_pose_nullspace(current_pose)
                # print(times)
                # rotation_times += times
                step += 1
                pre_pre_pre = pre_pre
                pre_pre = pre
                pre = score
                time.sleep(0.2)
                print(err_list)
                print(np.cumsum(err_list, axis=1))
        input(" ")
