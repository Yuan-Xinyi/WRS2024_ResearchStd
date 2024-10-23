import cobotta_x_new as cbtx
import config_file as conf
import time
import env_bulid_kobe as env
import copy
import numpy as np

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


tip_id = 0
robot_x = cbtx.CobottaX()

'''
        get chemical
        '''
move_result = False
# chemical_pos = env.deep_plate._hole_pos_list[tip_id]
chemical_pos = copy.deepcopy(env.microplate._hole_pos_list[0])
# gm.gen_sphere(chemical_pos).attach_to(base)
print(chemical_pos)
chemical_pos[:3] += conf.adjust_pos
chemical_pos[2] = conf.chemical_height

move_to_new_pose(chemical_pos)
time.sleep(0.1)
current_pose = robot_x.get_pose_values()
input("chemical pos 0 ")

chemical_pos = copy.deepcopy(env.microplate._hole_pos_list[6])
# gm.gen_sphere(chemical_pos).attach_to(base)
print(chemical_pos)
chemical_pos[:3] += conf.adjust_pos
chemical_pos[2] = conf.chemical_height

move_to_new_pose(chemical_pos)
input("chemical pos 6 ")



'''
insert chemical
'''
print("insert chemical")
temp_pose_chemical = robot_x.get_pose_values()
time.sleep(0.1)
insert_pose = copy.deepcopy(temp_pose_chemical)
robot_x.open_gripper()
insert_pose[2] -= conf.height_insert_chem
move_to_new_pose(insert_pose, 30)
# robot_x.defult_gripper(speed=70)
# robot_x.open_gripper()
robot_x.defult_gripper(dist=0.018, speed=50)
# robot_x.defult_gripper(speed=50)
# time.sleep(0.5)
move_to_new_pose(temp_pose_chemical)

'''
water plant
'''
input("water plant")
# balance_id = tip_id % 4 + 7
balance_id = 7
robot_x.move_pose(conf.execute_pose[0])
robot_x.move_pose(conf.execute_pose[balance_id])
print(f"position{balance_id} pose: np.{repr(robot_x.get_pose_values())}")
print(f"position{balance_id} jnt_values: np.{repr(robot_x.get_jnt_values())}")

robot_x.open_gripper()
time.sleep(0.5)
robot_x.defult_gripper()

'''
throw away tip
'''
input("throw away tip")
eject_jnt_values_list = conf.eject_jnt_values_list
eject_len = len(eject_jnt_values_list)
eject_jnt_values = eject_jnt_values_list[tip_id % eject_len]
# move_to_new_pose(eject_pose)
time.sleep(0.2)
robot_x.move_jnts(eject_jnt_values)

current_pose = robot_x.get_pose_values()
time.sleep(0.1)
current_pose[2] -= 0.05
move_to_new_pose(current_pose)
robot_x.close_gripper()
robot_x.defult_gripper()

tip_id += 1
# input(f"tip_id:{tip_id}")

time.sleep(0.1)