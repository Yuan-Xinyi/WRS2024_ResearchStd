import numpy as np
import wrs.basis.robot_math as rm

long_pipette_toggle = False
cobotta_base_toggle = True
if cobotta_base_toggle:
    base_height = 0.033
else:
    base_height = 0.01

# rack locator
init_obs_rot_list = [
    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
]

# init_obs_pos_list = [
#     np.array([.23, .03, .15+base_height]), np.array([.40, .03, .15+base_height]),
#     np.array([.40, .15, .145+base_height]), np.array([.23, .15, .15+base_height]),
# ]

init_obs_pos_list = [
    np.array([.21, .03, .22]), np.array([.35, .03, .22]),
    np.array([.35, .15, .22]), np.array([.21, .15, .22]),
]

calib_path = ["./fisheye_0_data.yaml", "./fisheye_1_data.yaml", "fisheye_stereo_data.yaml"]
model_path = {"resnet": "../models_learning/resnet_spiral_t_hex",
              # "vit": "../models_learning/vit_spiral_new_44",
              "vit": "../models_learning/vit_spiral_t_hex",
              "vit_long": "../models_learning/vit_spiral_1000_42",
              "vit_d405": "./trained_model/vit_120_best",
              "knn": ("../models_learning/autoencoder_3", "../models_learning/knn_model_new_5"),
              "svm": ("../models_learning/autoencoder_3", "../models_learning/svc_model_new")
              }

'''
    x+
 y+   y-
    x-
x<35  y<30
'''
# x0 y0 x1 y1
camera_pixel = np.array([9, 27, 20, 23])
long_pipette_height = 0
record_height = 0.204 - base_height
height_insert_tip = 0.007
height_insert_chem = 0.055
init_id = 0
open_id = 1
close_id = open_id + 3
balance_temp_id = close_id + 5
balance_id = balance_temp_id + 1
## 200ul
execute_pose = [np.array([0.24925295, -0.03881077, 0.23421913, 1.50819933, -0.02325565, 0.49082779, 5.]),
                # 0, init pose

                ## open door
                # 1
                np.array([0.22013102, 0.00740778, 0.13282731, 1.37417918, -1.40123147, 1.71041547, 5.]),  # 1,
                # 2
                np.array(
                    [0.22003292, -0.04365521, 0.12541219 - base_height, 1.7172739, -1.40196346, 1.00293653, 5.]),
                # 3
                np.array(
                    [0.34730411, -0.04495072, 0.13831297 - base_height, 2.89513745, -1.34484741, 0.08914309, 5.]),

                ## close door
                # 1
                np.array([4.02362679e-01, -2.64196087e-04, 1.40835937e-01, -1.82230148e+00,
                          -1.51924820e+00, -1.40942868e+00, 5.]),
                # 2,
                np.array([0.40391319, -0.06196657, 0.12052524, -1.9636749, -1.34779482,
                          -1.8042933, 5.]),
                # 3
                np.array([0.35358331, -0.10174771, 0.12224206, -1.73401699, -1.39851119,
                          -2.64436558, 5.]),
                # 4
                np.array([0.30006426, -0.12579298, 0.10982543, -2.26276096, -1.40272374,
                          -2.34483813, 5.]),

                # 5
                np.array([0.33709614, -0.03278703, 0.18053553, -1.66104367, -1.4895215,
                          -2.70891899, 5.]),

                ## eject pose
                # 1
                np.array([2.85968523e-01, -1.15305811e-01, 2.19812826e-01, 1.57079717e+00,
                          -3.62304183e-06, 5.87022900e-02, 5.00000000e+00]),
                # 2
                np.array([0.29447415, -0.11644559, 0.22971744, 1.23103486, 0.07083234,
                          0.16138103, 5.]),

                ]

# execute_jnt_values are based on execute_pose by P2J
execute_jnt_values = [np.array([0.2007242, 0.3515414, 1.55351247, 1.41295588, -1.32032907, 0.38656832]),
                      np.array([0.42610328, 0.68611512, 2.30673115, 0.48262829, -1.47188178, 2.90636221]),
                      np.array([0.23254648, 0.96384662, 2.09251101, 0.65902283, -1.47878824, 2.92236476]),
                      np.array([0.04872915, 1.00437266, 1.43025871, 0.33825634, -0.66812806, 2.86291929]),
                      np.array([0.1532776, 1.17283928, 0.90462523, 0.48228688, -0.54561093, 2.77391944]),
                      np.array([-0.03287257, 1.32610817, 0.47735233, 1.37481512, -0.59524534, 2.06861654]),
                      np.array([-0.06450369, 1.3258609, 0.4969952, 1.4945516, -1.17591029, 2.00249447]),
                      np.array([-0.15182907, 1.1678252, 0.84947815, 1.5545546, -1.28726829, 2.13534406]),
                      np.array([0.14674446, 1.0445695, 0.69527493, 1.54475481, -1.37639804, 1.82261003]),
                      np.array([-0.11307364, 0.7544349, 0.97835351, 1.54282392, -1.4012911, 0.16437222]),
                      np.array([-0.06098585, 0.82340254, 1.03814128, 1.17259104, -1.44451275, 0.28368126]), ]

eject_jnt_values1 = np.array([-0.02471361, 0.77401362, 1.26685552, 1.52887167, -1.31105645, -0.49606312])
# eject_jnt_values2 = np.array([-0.10698392, 1.11330726, 1.0179014, 1.64258665, -1.14497269, -0.76750369])
eject_jnt_values_list = [eject_jnt_values1]

height_rack = 0.062 - base_height - 0.008

recognize_height = record_height
chemical_height = 0.25 - base_height + long_pipette_height

is_spring_bed = False
if is_spring_bed:
    height_init = -base_height + 0.008

# hand2cam_mat = np.array([[0.05333406, -0.00250666, 0.99857358, 0.04810907],
#                          [-0.99830838, 0.02304735, 0.05337777, 0.01014368],
#                          [-0.02314826, -0.99973122, -0.00127321, -0.0297884],
#                          [0., 0., 0., 1.]])

hand2cam_mat = np.array([[0.99971071, -0.01540646, -0.01846945, -0.01483953],
                         [-0.01872506, -0.01661685, -0.99968658, -0.06694395],
                         [0.01509471, 0.99974321, -0.01690054, -0.03058389],
                         [0., 0., 0., 1.]])

adjust_pos = np.array([-0.001, 0.002, 0])

start_pose = np.array([0.14707305, 0.15072761, 0.27892955, 1.5541999, -0.05283379, 2.09194562, 5.])

# photo_jnt_values = np.array([-0.94987767, 0.18409651, 2.10643734, 0.81782603, -0.90152327, 0.11157466])
# photo_path = [np.array([-0.02678293, 0.73298144, 1.26838376, 1.53825622, -1.30767671, -0.53690161]),
#               np.array([0.07055423, 0.59481795, 1.43496736, 1.51135683, -1.33182243, -0.21693343]),
#               np.array([0.04430252, 0.43089145, 1.61801326, 1.5011152, -1.39190271, -0.05812972]),
#               np.array([0.01805081, 0.26696496, 1.80105916, 1.49087356, -1.451983, 0.10067399]),
#               np.array([-0.01695147, 0.0483963, 2.04512037, 1.47721805, -1.53209004, 0.31241227]),
#               np.array([-0.04245065, -0.11083109, 2.22291909, 1.46727, -1.59044807, 0.46666373]),
#               np.array([-0.04245065, -0.11083109, 2.22291909, 1.46727, -1.59044807, 0.46666373]),
#               np.array([-0.23757458, -0.04741284, 2.19787203, 1.32762014, -1.44230865, 0.39030897]),
#               np.array([-0.49773982, 0.03714482, 2.16447596, 1.14142032, -1.24478942, 0.28850262]),
#               np.array([-0.69286375, 0.10056307, 2.1394289, 1.00177045, -1.09665, 0.21214785]),
#               np.array([-0.94987767, 0.18409651, 2.10643734, 0.81782603, -0.90152327, 0.11157466])
#               ]
photo_start_jnts_values = np.array([-0.02678293, 0.73298144, 1.26838376, 1.53825622, -1.30767671, -0.53690161])
photo_temp_pose_values = np.array([0.1157, -0.088, 0.1521, 1.5992, 0.0229, 0.212, 5.])
photo_end_pose_values = np.array([8.95420335e-02, -2.43444129e-01, 1.90739357e-01, 1.54671300e+00,
                                  -2.63398253e-01, 4.46377745e-03, 5.])
photo_path_kobe = [np.array([-0.02678293, 0.73298144, 1.26838376, 1.53825622, -1.30767671, -0.53690161]),
                   np.array([0.07055423, 0.59481795, 1.43496736, 1.51135683, -1.33182243, -0.21693343]),
                   np.array([0.04430252, 0.43089145, 1.61801326, 1.5011152, -1.39190271, -0.05812972]),
                   np.array([0.01805081, 0.26696496, 1.80105916, 1.49087356, -1.451983, 0.10067399]),
                   np.array([-0.01695147, 0.0483963, 2.04512037, 1.47721805, -1.53209004, 0.31241227]),
                   np.array([-0.04245065, -0.11083109, 2.22291909, 1.46727, -1.59044807, 0.46666373]),
                   np.array([-0.04245065, -0.11083109, 2.22291909, 1.46727, -1.59044807, 0.46666373]),
                   np.array([-0.23757458, -0.04741284, 2.19787203, 1.32762014, -1.44230865, 0.39030897]),
                   np.array([-0.49773982, 0.03714482, 2.16447596, 1.14142032, -1.24478942, 0.28850262]),
                   np.array([-0.69286375, 0.10056307, 2.1394289, 1.00177045, -1.09665, 0.21214785]),
                   np.array([-0.94987767, 0.18409651, 2.10643734, 0.81782603, -0.90152327, 0.11157466]),
                   np.array([-0.93622015, 0.48257906, 1.95461969, 0.80746565, -1.06648045, 0.01512216]),

                   ]

# microplate_pose = np.array([0.21, 0.223, 2.21666477e-01, 1.57079297e+00,
#                             -3.45501132e-06, 1.87842781e+00, 5.00000000e+00])

# microplate_pose_record = np.array([.41, -0.075, 1.87526343e-01, 1.57097521e+00, 4.24832910e-05, 1.35055132e+00, 5.])
# microplate_rotangle = np.pi

microplate_pose_record = np.array([0.26, -0.08, 1.91596861e-01, 1.57123192e+00, 0, 1.45631612e+00, 5.0])
microplate_rotangle = np.pi / 2

pos_y0 = .0315
pos_x0 = .0495

microplate_pos = np.array(
    [microplate_pose_record[0] + pos_x0 - 0.003, microplate_pose_record[1] + pos_y0, -base_height])
microplate_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), microplate_rotangle)
microplate_homo = rm.homomat_from_posrot(microplate_pos, microplate_rot)
microplate_pose = [microplate_pos, microplate_rot]

OBS_HEIGHT = .025
OBS_HEIGHT_REFINE = .15
PINK_RACK_PATH = "./meshes/rack_violamo.stl"
PINK_RACK_HEIGHT = .049
PINK_HEIGHT_RANGE = (.052, 0.065)
PINK_HEIGHT_RANGE_REFINE = (.053, 0.064)

PURPLE_RACK_PATH = "./meshes/rack_mbp.stl"
PURPLE_RACK_HEIGHT = .037
PURPLE_HEIGHT_RANGE = (.036 - .004, 0.047 - .008)

# WORK_RACK_PATH = PINK_RACK_PATH
# WORK_HEIGHT_RANGE = PINK_HEIGHT_RANGE
# WORK_RACK_HEIGHT = PINK_RACK_HEIGHT
# WORK_RACK_HEIGHT_REFINE = PINK_HEIGHT_RANGE_REFINE

WORK_RACK_PATH = PURPLE_RACK_PATH
WORK_HEIGHT_RANGE = PURPLE_HEIGHT_RANGE
WORK_RACK_HEIGHT = PURPLE_RACK_HEIGHT
WORK_RACK_HEIGHT_REFINE = PURPLE_HEIGHT_RANGE

rack_transform = np.array([[-7.18623300e-01, 6.95389440e-01, 3.75221802e-03,
                            3.29077027e-01],
                           [-6.95393760e-01, -7.18628886e-01, 2.07760259e-04,
                            1.06219507e-01],
                           [2.84092654e-03, -2.45996763e-03, 9.99992939e-01,
                            -3.36969206e-02],
                           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                            1.00000000e+00]])

init_joint_values = np.radians(np.asarray([10.0, -10.0, 100.0, 80.0, 0.0, 100.0]))
