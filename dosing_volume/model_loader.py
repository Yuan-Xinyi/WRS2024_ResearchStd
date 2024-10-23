import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from vit_pytorch.efficient import ViT
from linformer import Linformer
from PIL import Image
import numpy as np
import wrs.robot_sim.robots.cobotta.cobotta as cbts
import wrs.basis.robot_math as rm
import math
import time


def spiral(num):
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


x_list, y_list = spiral(200)
spiral_list = np.zeros((200, 2))
spiral_list.T[0] = x_list
spiral_list.T[1] = y_list


def get_rbt_gl_from_pipette(pipette_gl_pos, pipette_gl_angle=0):
    dist = 0.007
    pipette_gl_rot = rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(pipette_gl_angle))
    pipette_gl_mat = rm.homomat_from_posrot(pipette_gl_pos, pipette_gl_rot)
    pipette_tcp_pos = np.array([-0.008, -0.15485, 0.01075]) + np.array([0.0015, -dist, -0.0058])
    pipette_tcp_rot = np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), -math.pi / 2),
                             rm.rotmat_from_axangle(np.array([0, 1, 0]), -math.pi / 2))
    pipette_tcp_mat = rm.homomat_from_posrot(pipette_tcp_pos, pipette_tcp_rot)
    rbt_tcp_pos = np.array([0, 4.7, 10]) / 1000
    rbt_tcp_mat = rm.homomat_from_posrot(rbt_tcp_pos, np.eye(3))
    rbt_gl_mat = np.dot(np.dot(pipette_gl_mat, np.linalg.inv(pipette_tcp_mat)), rbt_tcp_mat)
    rot_euler = rm.rotmat_to_euler(rbt_gl_mat[:3, :3])
    return np.append(np.append(rbt_gl_mat[:3, 3], rot_euler), 5)


class ResnetModel(object):
    def __init__(self, model_path, device="cpu"):
        self.model_resnet50 = self.model_load(model_path, device)
        self.pic_transformer = transforms.Compose([transforms.ToTensor()])

    def model_load(self, path, device):
        new_model = torchvision.models.resnet50(weights=None).to(device)
        new_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        new_model.eval()
        return new_model

    def get_score(self, pic, show_img=False):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0)
        model = self.model_resnet50
        [val_output] = model(pic_tensor).detach().numpy()
        if show_img:
            def show_img_callback():
                img = np.array(transforms.ToPILImage()(pic_tensor.squeeze(0)))
                cv2.imshow("img", img)
                cv2.waitKey(0)
        else:
            show_img_callback = lambda: None
        # print(val_output)
        return val_output.argmax(), val_output, show_img_callback


class TransformerModel(object):
    def __init__(self, model_path, img_size=(45, 80), patch_size=5, num_classes=145, dim=256, img_transformer=None,
                 device="cpu"):
        self.model_vit = self.model_load(model_path, img_size, patch_size, num_classes, dim, device)
        if img_transformer is None:
            self.pic_transformer = transforms.Compose([transforms.ToTensor()])
        else:
            self.pic_transformer = img_transformer
        self.device = device

    def model_load(self, path, img_size, patch_size, num_classes, dim, device):
        efficient_transformer = Linformer(dim=dim,
                                          seq_len=int(img_size[0] / patch_size * img_size[1] / patch_size) + 1,
                                          # n*m patches + 1 cls-token
                                          depth=12,
                                          heads=8,
                                          k=64)
        new_model = ViT(dim=dim,
                        image_size=img_size,
                        patch_size=patch_size,
                        num_classes=num_classes,
                        transformer=efficient_transformer,
                        channels=3).to(device)
        new_model.load_state_dict(torch.load(path, map_location=torch.device(device)))
        new_model.eval()
        self.device = device
        return new_model

    def get_score(self, pic, show_img=False):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0).to(self.device)
        model = self.model_vit
        if self.device == "cpu":
            [val_output] = model(pic_tensor).detach().numpy()
        else:
            [val_output] = model(pic_tensor).cpu().detach().numpy()
        if show_img:
            def show_img_callback():
                img = np.array(transforms.ToPILImage()(pic_tensor.squeeze(0)))
                cv2.imshow("img", img)
                cv2.waitKey(0)
        else:
            show_img_callback = lambda: None
        # print(val_output)
        return val_output.argmax(), val_output, show_img_callback



class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                     nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                     nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(2, 2),

                                     nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                     nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(True))

        self.decoder = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1))

        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.ReLU(True),
                                      nn.ConvTranspose2d(3, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                      nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.decoder2(x)
        return x


class SciModel(object):
    def __init__(self, model_path_ae, model_path_knn, img_size=(45, 80), device="cpu"):
        self.img_size = img_size
        self.model_loader(model_path_ae, model_path_knn, device)
        self.pic_transformer = transforms.Compose([transforms.Resize((48, img_size[1])), transforms.ToTensor()])

    def model_loader(self, path_ae, path_sci, device):
        self.model_ae = Autoencoder().to(device)
        self.model_ae.load_state_dict(torch.load(path_ae, map_location=torch.device(device)))
        self.model_ae.eval()
        self.model_sci = torch.load(path_sci, map_location=torch.device(device))

    def get_score(self, pic):
        pic = Image.fromarray(pic)
        pic_tensor = self.pic_transformer(pic)
        pic_tensor = pic_tensor.unsqueeze(0)
        pic_feature = self.model_ae.encoder(pic_tensor)
        pic_feature = self.model_ae.decoder(pic_feature)
        pic_recover = self.model_ae.decoder2(pic_feature)
        pic_feature_arr = pic_feature.view(-1, self.img_size[1] * 9).detach().numpy()
        score = self.model_sci.predict(pic_feature_arr)
        return score[0], pic_recover[0].permute(1, 2, 0).detach().numpy()


# class MaskRCNNModel(object):
#     def __init__(self, model_path, class_num=2):


if __name__ == "__main__":

    import fisheye_camera as camera
    import config_file as conf

    tic = time.time()
    # model_row_s = TransformerModel("tip_cam_spiral_c", (45, 80), 5, 165)
    model_knn = SciModel("../models_learning/autoencoder_3", "../models_learning/knn_model_new_5")
    model_svm = SciModel("../models_learning/autoencoder_0", "../models_learning/svc_model_new")
    # model_row_tip = GetDirection("tip_cam_spiral", (45, 80), 5, 145)
    print("model:", time.time() - tic)
    tic = time.time()
    fcam = camera.FisheyeCam(conf.calib_path)
    print("camera:", time.time() - tic)
    tic = time.time()
    robot_s = cbts.Cobotta()
    # rt_sys = cbtrt.CobottaRTClient(server_ip="localhost")
    print("connected")

    # current_pose = rt_sys.get_pose_values()
    # time.sleep(0.1)
    # current_pose[2] -= 0.01
    # rt_sys.rbt_move_p(current_pose)
    # current_pose = rt_sys.get_pose_values()
    # print(current_pose)
    # current_jnts = rt_sys.get_jnts_value()
    # print(np.degrees(current_jnts))
    # robot_s.fk(jnt_values=current_jnts)
    # tcp_pos,tcp_rot = robot_s.get_gl_tcp("arm")
    # print(tcp_pos)
    # print(tcp_pos-np.dot(current_pose[:3],tcp_rot))
    tgt_pos = np.array([0.232, 0.023, 0.064])
    rot_angle = 1
    start_pose_values = get_rbt_gl_from_pipette(tgt_pos, rot_angle)
    print(start_pose_values)
    # rt_sys.rbt_move_p(start_pose_values)
    # time.sleep(0.5)
    # input("c")

    # # for pic in range(1):
    # sample_pos_list = get_sample_pose_values_list(tgt_pos,0,theta=np.radians(45))
    # # rot_angle -= 3
    # for pos in range(len(sample_pos_list)):
    #     rt_sys.send_pose_values(sample_pos_list[pos])
    # # time.sleep(0.2)
    # rt_sys.start_sample()
    #

    while 1:
        # current_pose = rt_sys.get_pose_values()
        pic = fcam.get_frame_cut_combine_row()
        cv2.imshow("pic", pic)
        # direct_row_s, weight_row_s, show_img_callback = model_row_s.get_score(pic)
        score_knn, pic_feature_knn = model_knn.get_score(pic)
        score_svm, pic_feature_svm = model_svm.get_score(pic)
        print(pic_feature_knn.shape)
        cv2.imshow("ae", pic_feature_knn)
        print(score_knn)
        cv2.waitKey(0)
    #
    #     rt_sys.insert()
    #     # time.sleep(0.5)
    #     rt_sys.abandon_tip()

    # rt_sys.close_connection()
