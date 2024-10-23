import os
import json
import time
from typing import Literal

import numpy as np

from wrs.drivers.devices.dynamixel_sdk.sdk_wrapper import DynamixelMotor, PortHandler

# get the path of the current file
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DATA_FILE_NAME = os.path.join(THIS_DIR, "resources/gripper_x_calib_data.json")
CALIBRATION_DATA_FILE_NAME_PIPETTE = os.path.join(THIS_DIR, "resources/pipette_x_calib_data.json")
SAVE_CALIBRATION_DATETIME = "LARGEHAND_GRIPPER_X_SAVE_CALIBRATION_DATETIME"
IS_CALIBRATED_NAME = "LARGEHAND_GRIPPER_X_IS_CALIBRATED"
CLOSE_GRIPPER_MOTOR_POS_NAME = "LARGEHAND_GRIPPER_X_CLOSE_GRIPPER_MOTOR_POS"
OPEN_GRIPPER_MOTOR_POS_NAME = "LARGEHAND_GRIPPER_X_OPEN_GRIPPER_MOTOR_POS"


class LargehandGripperX(object):
    def __init__(self,
                 device: str = "COM7",
                 motor_ids: tuple or list = (1, 2),
                 motor_model: str ="",
                 baudrate: int = 115200,
                 port_handler=None,
                 op_mode: int = 5,
                 gripper_limit: tuple or list = (0, 0.0144),
                 motor_max_current =200,
                 load_calibration_data: bool = True,
                 calib_data_path=None,
                 close_gripper_direction=-1):
        """
        Initialize the Nova2 gripper
        :param device: The device name of the serial port
        :param motor_ids: dynaxmiel motor ids of the gripper
        :param baudrate: The baudrate of the serial port
        :param op_mode: The operation mode of the dynamixel motor
                    0	Current Control Mode
                    1	Velocity Control Mode
                    3	Position Control Mode
                    4   Extended Position Control Mode (Multi-turn)
                    5   Current-based Position Control Mode
                    16  PWM Control Mode
        :param gripper_limit: The gripper limit, (min, max)
        :param motor_max_current: The max current of the motor (See the manual for details)
        :param load_calibration_data: If True, load the calibration data from the file
        """
        assert motor_ids is not None, "Please specify the motor ids."
        assert isinstance(motor_ids, (tuple, list)), "The motor ids should be a tuple or list."
        assert gripper_limit is not None, "Please specify the gripper limit."
        assert isinstance(gripper_limit, (tuple, list)), "The gripper limit should be a tuple or list."
        assert isinstance(baudrate, int), "The baudrate should be an integer."
        self._motor_x = DynamixelMotor(device=device, baud_rate=baudrate, port_handler=port_handler)
        self._motor_ids = motor_ids
        # Set the operation mode
        for motor_id in self.motor_ids:
            if self._motor_x.get_dxl_op_mode(motor_id) != op_mode:
                self._motor_x.disable_dxl_torque(motor_id)
                self._motor_x.set_dxl_op_mode(op_mode=op_mode, dxl_id=motor_id)
        # Enable the torque
        [self._motor_x.enable_dxl_torque(motor_id) for motor_id in motor_ids]
        # Set the current limit
        self._current_max = motor_max_current
        [self._motor_x.set_dxl_goal_current(motor_max_current, motor_id) for motor_id in motor_ids]
        # Set the gripper limit
        self._gripper_limit = gripper_limit
        # read the calibration data
        self.is_calibrated = False
        self._close_gripper_motor_pos = None
        self._open_gripper_motor_pos = None
        if calib_data_path is None:
            self.calib_data_path = CALIBRATION_DATA_FILE_NAME
        else:
            self.calib_data_path = calib_data_path
        if load_calibration_data and os.path.exists(self.calib_data_path):
            with open(calib_data_path, "r") as f:
                env = json.load(f)
                if SAVE_CALIBRATION_DATETIME in env:
                    save_calibration_datetime = env[SAVE_CALIBRATION_DATETIME]
                if time.time() - save_calibration_datetime < 3600 * 24 * 1:  # 1 days to update the calibration data
                    if IS_CALIBRATED_NAME in env:
                        self.is_calibrated = env[IS_CALIBRATED_NAME]
                    if CLOSE_GRIPPER_MOTOR_POS_NAME in env:
                        self._close_gripper_motor_pos = env[CLOSE_GRIPPER_MOTOR_POS_NAME]
                    if OPEN_GRIPPER_MOTOR_POS_NAME in env:
                        self._open_gripper_motor_pos = env[OPEN_GRIPPER_MOTOR_POS_NAME]

    @property
    def motor_ids(self):
        return self._motor_ids

    @property
    def is_calibrated(self):
        return self._is_calibrated

    @is_calibrated.setter
    def is_calibrated(self, value: bool):
        assert isinstance(value, bool), "The value should be a boolean."
        self._is_calibrated = value

    # TODO write the equation to exactly calucate the relationship between the gripper width and the motor position
    def map_gripper_width_to_motor_pos(self, width: float or int) -> list:
        """
        Map the gripper width to the motor position
        :param width: The width of the gripper
        :return: The motor position
        """
        # assert self.is_calibrated, "Please calibrate the gripper first."
        assert isinstance(width, (float, int)), "The width should be a number."
        assert self._gripper_limit[0] - 0.01 <= width <= self._gripper_limit[
            1] + 0.01, "The width is out of the gripper limit."
        # Calculate the motor position
        motor_pos = [int(self._close_gripper_motor_pos[i] + (
                self._open_gripper_motor_pos[i] - self._close_gripper_motor_pos[i]) * width / (
                                 self._gripper_limit[1] - self._gripper_limit[0]))
                     for i, motor_id in enumerate(self.motor_ids)]
        return motor_pos

    def calibrate(self, close_gripper_direction: Literal[-1, 1] = 1, speed: int = 100):
        assert close_gripper_direction in [-1, 1], "The close_gripper_direction should be -1 or 1."
        if self._motor_x.get_dxl_op_mode(self.motor_ids[0]) == 3:
            motor_MAX = self._motor_x._control_table.DXL_MAX_POSITION_VAL
            motor_MIN = self._motor_x._control_table.DXL_MIN_POSITION_VAL
        else:
            motor_MAX = 1048575
            motor_MIN = 0
        # Set the speed of the motor
        [self._motor_x.set_dxl_position_p_gain(speed, motor_id) for motor_id in self.motor_ids]
        time.sleep(.1)
        # Set the goal current of the gripper
        [self._motor_x.set_dxl_goal_current(200, motor_id) for motor_id in self.motor_ids]
        time.sleep(.1)
        motor_locs = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self.motor_ids]
        if close_gripper_direction == 1:
            # close the gripper

            [self._motor_x.set_dxl_goal_pos(
                min(motor_locs[i] + 4000, motor_MAX)
                , motor_id) for i, motor_id in enumerate(self.motor_ids)]
        else:
            [self._motor_x.set_dxl_goal_pos(
                max(motor_locs[i] - 4000, motor_MIN)
                , motor_id) for i, motor_id in enumerate(self.motor_ids)]
        time.sleep(.1)
        while np.any([self._motor_x.is_moving(motor_id) for motor_id in self.motor_ids]):
            time.sleep(.1)
        # get the current position
        motor_close_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self.motor_ids]
        time.sleep(.5)
        # Get the open position
        if close_gripper_direction == 1:
            # close the gripper
            [self._motor_x.set_dxl_goal_pos(max(motor_locs[i] - 4000, motor_MIN)
                                            , motor_id) for i, motor_id in enumerate(self.motor_ids)]
        else:
            [self._motor_x.set_dxl_goal_pos(min(motor_locs[i] + 4000, motor_MAX)
                                            , motor_id) for i, motor_id in enumerate(self.motor_ids)]
        time.sleep(.1)
        while np.any([self._motor_x.is_moving(motor_id) for motor_id in self.motor_ids]):
            time.sleep(.1)
        # get the current position
        motor_open_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self.motor_ids]
        self._close_gripper_motor_pos = motor_close_pos
        self._open_gripper_motor_pos = motor_open_pos
        self.is_calibrated = True  # set the calibration flag
        self.save_calibration_data()

    def save_calibration_data(self):
        """
        Save the calibration data to the file
        """
        data = dict()
        data[SAVE_CALIBRATION_DATETIME] = time.time()
        data[IS_CALIBRATED_NAME] = self.is_calibrated
        data[CLOSE_GRIPPER_MOTOR_POS_NAME] = self._close_gripper_motor_pos
        data[OPEN_GRIPPER_MOTOR_POS_NAME] = self._open_gripper_motor_pos
        with open(self.calib_data_path, "w") as f:
            json.dump(data, f, indent=4)

    def set_direction_speed(self, direct=1, speed: int = 100, wait: bool = True):
        print("speed control")
        print([self._motor_x.get_dxl_op_mode(motor_id) for motor_id in self.motor_ids])
        # [self._motor_x.set_dxl_op_mode(1, motor_id) for motor_id in self.motor_ids]
        vel = int(direct * speed / 100 * 200)
        print(vel)
        [self._motor_x.set_dxl_goal_vel(vel, motor_id) for motor_id in self.motor_ids]
        if wait:
            time.sleep(.1)
            while np.any([self._motor_x.is_moving(motor_id) for motor_id in self.motor_ids]):
                time.sleep(.1)


    def set_gripper_width(self, width: float or int, speed: int = None, grasp_force: int = 100,
                          wait: bool = True, disable_motor_id: int = None) -> bool:
        """
        Set the gripper width
        :param width: The width of the gripper
        :param speed: The speed of the gripper
        :param grasp_force: The grasp force of the gripper (Described by current)
        :param wait: If True, wait until the gripper is stopped
        :return: True if the gripper is stopped
        """
        if not self.is_calibrated:
            raise ValueError("Please calibrate the gripper first.")
        assert isinstance(width, (float, int)), "The width should be a number."
        assert 0 < grasp_force <= 100, "The grasp force should between 1 and 100."
        assert self._gripper_limit[0] - 0.01 <= width <= self._gripper_limit[
            1] + 0.01, "The width is out of the gripper limit."
        # Set the speed of the gripper
        if speed is not None:
            [self._motor_x.set_dxl_op_mode(1, motor_id) for motor_id in self.motor_ids]
            [self._motor_x.set_dxl_goal_vel(speed, motor_id) for motor_id in self.motor_ids]
            time.sleep(.1)
        else:
            [self._motor_x.set_dxl_op_mode(5, motor_id) for motor_id in self.motor_ids]
        # Set the goal current of the gripper
        [self._motor_x.set_dxl_goal_current(int(grasp_force / 100 * self._current_max), motor_id) for motor_id in
         self.motor_ids]
        time.sleep(.1)
        ret = [self._motor_x.set_dxl_goal_pos(self.map_gripper_width_to_motor_pos(width)[i], self.motor_ids[i])
               for i, motor_id in enumerate(self.motor_ids) if disable_motor_id != motor_id]
        if wait:
            time.sleep(.1)
            while np.any([self._motor_x.is_moving(motor_id) for motor_id in self.motor_ids]):
                time.sleep(.1)
        return np.all(ret)

    def get_gripper_width(self) -> float:
        """
        Get the gripper width
        :return: The width of the gripper
        """
        if not self.is_calibrated:
            raise ValueError("Please calibrate the gripper first.")
        motor_pos = [self._motor_x.get_dxl_pos(motor_id) for motor_id in self.motor_ids]
        width = (motor_pos[0] - self._close_gripper_motor_pos[0]) / (
                self._open_gripper_motor_pos[0] - self._close_gripper_motor_pos[0]) * (
                        self._gripper_limit[1] - self._gripper_limit[0]) + self._gripper_limit[0]
        return width

    def open_gripper(self, speed: int = None, grasp_force=100, wait: bool = True, disable_motor_id: int = None) -> bool:
        """
        Open the gripper
        """
        self.disable_torque()
        self.enable_torque()
        return self.set_gripper_width(self._gripper_limit[1] + 0.01, speed, grasp_force, wait, disable_motor_id)

    def close_gripper(self, speed: int = None, grasp_force=100, wait=True, disable_motor_id: int = None) -> bool:
        """
        Close the gripper
        """
        return self.set_gripper_width(self._gripper_limit[0] - 0.01, speed, grasp_force, wait, disable_motor_id)

    def init_gripper(self, init_persent=0.7, speed: int = None, grasp_force=170, wait=True,
                     disable_motor_id: int = None) -> bool:
        init_pose = (self._gripper_limit[1] - self._gripper_limit[0]) * init_persent
        return self.set_gripper_width(init_pose, speed, grasp_force, wait, disable_motor_id)

    def disable_torque(self, dxl_ids=None):
        if dxl_ids is None:
            dxl_ids = self.motor_ids
        for motor_id in dxl_ids:
            self._motor_x.disable_dxl_torque(motor_id)

    def enable_torque(self, dxl_ids=None):
        if dxl_ids is None:
            dxl_ids = self.motor_ids
        for motor_id in dxl_ids:
            self._motor_x.enable_dxl_torque(motor_id)

    def reboot(self, dxl_ids=None):
        if dxl_ids is None:
            dxl_ids = self.motor_ids
        for motor_id in dxl_ids:
            self._motor_x.rebot_dxl(motor_id)

    # def close_gripper_recording(self, motor_id, speed: int = 100, grasp_force=170, ):
    #     self._motor_x.set_dxl_position_p_gain(speed, motor_id)
    #     time.sleep(.1)
    #     # Set the goal current of the gripper
    #     self._motor_x.set_dxl_goal_current(grasp_force, motor_id)
    #     time.sleep(.1)
    #     ret = self._motor_x.set_dxl_goal_pos(self._close_gripper_motor_pos, motor_id)
    #     time.sleep(.1)
    #
    #     pos_list = []
    #     current_list = []
    #     while self._motor_x.is_moving(motor_id):
    #         time.sleep(.01)
    #         pos_list.append(self._motor_x.get_dxl_pos(motor_id))
    #         current_list.append(self._motor_x.get_dxl_current(motor_id))
    #
    #     return pos_list, current_list

    def get_dxl_pos(self):
        return [self._motor_x.get_dxl_pos(motor_id) for motor_id in self.motor_ids]
    # def __del__(self):
    #     """
    #     Disable the torque of the motor
    #     :return: None
    #     """
    #     try:
    #         [self._motor_x.disable_dxl_torque(motor_id) for motor_id in self.motor_ids]
    #     except Exception as e:
    #         print(e)


if __name__ == "__main__":
    device = 'COM4'
    baudrate = 57600
    port_handler = PortHandler(device)
    gripper_id = 10
    motor_id = 5
    gripper_calib_data_path = "resources/dynamixel_calib/gripper_x_calib_data"
    motor_calib_data_path = "./resources/dynamixel_calib/motor_x_calib_data"

    # gripper
    gp_x = LargehandGripperX(device=device, motor_ids=[gripper_id], baudrate=baudrate, port_handler=port_handler, op_mode=5,
                             gripper_limit=(0, 0.134),
                             motor_max_current=100, calib_data_path=gripper_calib_data_path)

    # if gp_x.is_calibrated is False:
    #     gp_x.calibrate(close_gripper_direction=1)

    gp_x.open_gripper()
