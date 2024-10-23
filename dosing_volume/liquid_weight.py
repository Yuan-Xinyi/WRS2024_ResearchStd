import copy
import os
import cv2
from cv2 import aruco
import time
import numpy as np
import glob
from torchvision import transforms
import model_loader as ml
import file_sys as fs
from realsensecrop import RealSenseD405Crop, letterbox
from wrs import rm

# for i in range(10):
#     if not os.path.exists(f"./num_recog/model/num_{i}/"):
#         os.mkdir(f"./num_recog/model/num_{i}/")


def get_mouse_position(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)








crop_weight = fs.load_json('resources/crop_weight.json')
def get_LCD_image_aruco(rs_pipeline,debug_toggle = False):
    img_input = rs_pipeline.get_color_img()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = aruco.DetectorParameters()
    if debug_toggle:
        cv2.imshow("img_input",img_input)
        # cv2.waitKey(0)
    time.sleep(1)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_input, aruco_dict, parameters=parameters)
    direction_axis = corners[np.argmax(ids)][0][0] - corners[np.argmin(ids)][0][1]
    rotate_angle = np.degrees(rm.angle_between_2d_vecs(direction_axis, np.array([-1, 0]))) - 3

    h, w = img_input.shape[:2]
    rotate_center = (w / 2, h / 2)
    rotate_matrix = cv2.getRotationMatrix2D(rotate_center, -rotate_angle, 1.0)
    balance_img_rotated = cv2.warpAffine(img_input, rotate_matrix, (w, h))
    corners_new, ids_new, rejectedImgPoints = aruco.detectMarkers(balance_img_rotated, aruco_dict,
                                                                  parameters=parameters)
    pos_1 = corners_new[np.argmax(ids_new)][0][1].astype(int)
    pos_2 = (0.5 * (corners_new[np.argmin(ids_new)][0][1] + corners_new[np.argmin(ids_new)][0][0])).astype(int)
    len_real = int(np.abs(pos_2[0] - pos_1[0]) * .93)
    height_real = int(130 * len_real / 295 * .93)
    print(len_real, height_real)
    pos_real = copy.deepcopy(pos_1)
    pos_real[0] += 20
    pos_real[1] += 30
    balance_img_crop = rs_pipeline.crop_img(balance_img_rotated, pos_real, (len_real, height_real))
    balance_img_crop = letterbox(balance_img_crop, crop_weight["img_size"][::-1], auto=False)[0]
    if debug_toggle:
        balance_img_with_marker = aruco.drawDetectedMarkers(balance_img_rotated.copy(), corners_new, ids_new)
        cv2.circle(balance_img_with_marker, pos_1, 20, (200, 100, 100))
        cv2.circle(balance_img_with_marker, pos_2, 20, (200, 100, 100))
        print(pos_1, pos_2)
        cv2.imshow("with marker", balance_img_with_marker)
        cv2.imshow(f"calib_axis", balance_img_rotated)
        cv2.imshow(f"calib_axis_crop", balance_img_crop)
    return balance_img_crop

def num_recog(img, model_num,adjust_data, toggle_debug=False):
    # color_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)[890:1020, 290:585]
    color_img = copy.deepcopy(img)
    wide = 44.5
    height = -0.5
    left = adjust_data[0]
    top = adjust_data[1]
    button = top+90
    img_separate = []
    img_show = copy.deepcopy(color_img)
    for i in range(5):
        img_num = color_img[int(top + height * i):int(button + height * i),
                  np.floor(left + wide * i).astype(int):np.ceil(left + wide * (i + 1)).astype(int)]
        img_separate.append(img_num)
        cv2.rectangle(img_show, (int(left + wide * i), int(button + height * i)),
                      (int(left + wide * (i + 1)), int(top + height * i)), [60, 40, 80])
    index_list = []
    np.set_printoptions(precision=3, suppress=True)
    for img in img_separate:
        index = model_num.get_score(img)
        index_list.append(index[0])
    print(index_list)
    if toggle_debug:
        cv2.imshow("img_debug", img_show)
        cv2.waitKey(1)
    return index_list

def get_weight_from_str(str_list):
    weight = 0
    for id,num in enumerate(str_list):
        weight+=int(num)/(10**id)
    print(weight)
    return weight

if __name__ == "__main__":
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", get_mouse_position)

    model_num = ml.TransformerModel(model_path="./trained_model/num_model_ver2",
                                    img_size=(200,100),
                                    patch_size=10,
                                    num_classes=10,
                                    dim=128,
                                    img_transformer=transforms.Compose([transforms.Resize((200,100)),transforms.ToTensor()]))

    rs_pipeline = RealSenseD405Crop()


    toggle_debug = True
    name = "test"
    img_father_list = glob.glob(f"./data/liquid_weight/weight_img/{name}_?/")
    # color_img = rs_pipeline.get_color_img()

    dosing_volume = []

    while True:
        balance_img = rs_pipeline.get_color_img()
        crop_weight = fs.load_json('resources/crop_weight.json')
        balance_img_rotate = cv2.rotate(balance_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # color_img = rs_pipeline.crop_img(balance_img_rotate,crop_weight["left_top"],crop_weight["img_size"])
        color_img = rs_pipeline.crop_img(balance_img_rotate, crop_weight["left_top"], np.floor(np.array(crop_weight["img_size"])*1.034).astype(int))
        color_img = letterbox(color_img,new_shape=(130,295),auto=False)[0]
        cv2.imshow("img",color_img)
        # color_img=get_LCD_image_aruco(rs_pipeline,debug_toggle=False)
        num_list = num_recog(color_img, model_num,crop_weight["left_top_p"], toggle_debug=True)
        # print(f"Weight: {num_list[0]}.{num_list[1]}{num_list[2]}{num_list[3]}{num_list[4]}g")
        cv2.waitKey(1)

    get_LCD_image_aruco(rs_pipeline,debug_toggle=True)
    cv2.waitKey(0)