
# -*- coding: utf-8 -*-

from copy import copy
import string
import cv2
import argparse
from model.model_64cls import Model
from distribution_analyzer import DistributionAnalyzer

import os
import sys
import random
import cv2
import numpy as np
from tqdm import tqdm
import gxipy as gx

import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities  
from webdriver_manager.chrome import ChromeDriverManager

from bilibili import BiliBili

class Target:
    def __init__(self):
        self.cls = 0
        self.pts = np.array([4, 2])

def getImgLists(path, default_img_format):
    img_list = []
    # Generate img list to travel
    for root, dirs, files in os.walk(path, topdown=False):
        for file_name in tqdm(files):
            # Continue while the file is illegal.
            if (file_name.endswith(extend) for extend in default_img_format):
                img = cv2.imread(os.path.join(path, file_name))
                img_list.append(img)
            else:
                continue
    return img_list

def get_real_url(rid):
    try:
        bilibili = BiliBili(rid)
        return True, bilibili.get_real_url()
    except Exception as e:
        print('Exception：', e)
        return False, ""

def getLiveURL(url, timeout=30, driver_path="chrome/chromedriver"):
    rid = url.split("/")[-1]
    ret, real_url = get_real_url(rid)
    if ret:
        print(real_url)
        print("Live URL 已获取: {}".format(real_url['线路1']))
        return real_url['线路1']
    else:
        print("Invalid URL, Retrying!")
        return getLiveURL(url,timeout,driver_path)

def vis_result(img, result, vis_ms=1):
    dst = copy(img)
    for det in result:
        for i in range(4):
            cv2.line(dst, tuple(det.pts[i % 4]), tuple(
                det.pts[(i + 1) % 4]), (0, 255, 0))
    
        color_str = "BRNP"
        id_str = "G12345OB"
        sz_str = "sb"
        text = "{}{}{}".format(color_str[det.color // 2],
                                id_str[det.id],
                                sz_str[det.color % 2])
        cv2.putText(dst,
                    text,
                    (det.pts[:, :][0][0] - 10 , det.pts[:, :][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,0),
                    thickness=1)
        cv2.putText(dst,
                    "{:.2f}".format(det.conf.data),
                    (det.pts[:, :][3][0] - 10 , det.pts[:, :][3][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,255,0),
                    thickness=1)
    cv2.imshow("Image", dst)
    cv2.waitKey(vis_ms)


def get36ClsFrom64Cls(det):
    target = Target()
    target.cls = 9 * (det.color // 2) + det.id
    target.pts = np.array(det.pts)
    return target

class Annotator():
    def __init__(self) -> None:        
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument("--type", "-t", type=str, help="输入类型,支持daheng,local_imgs, local_video, stream.", default="local_video")
        self.parser.add_argument("--input_dir", "-i", help="目标数据位置.")
        self.parser.add_argument("--video", help="目标视频位置.", default="/home/rangeronmars/Videos/Sentry/2022-06-25_16_13_43.avi")
        self.parser.add_argument("--dst_dir", help="数据保存位置.", default="labels")
        self.args = self.parser.parse_args()
        
        self.model = Model("./model/model_64cls.onnx")
        self.analyzer = DistributionAnalyzer()
        self.default_img_format = [".jpg", ".png", ".jpeg"]
        self.mean_exposure = 7500
        self.max_delta_exposure = 2500
        self.using_daheng = False
        self.using_local_imgs = False
        self.using_local_video = False
        self.using_stream = False
        self.type = self.args.type
        self.img_lists = []
        self.video_pth = self.args.video
        self.dst_dir = self.args.dst_dir
        self.cap = []

    def autoLabelByStream(self):
        """使用视频流自动标注
        """
        # -----------------------初始化-----------------------------
        cnt = 0
        frames_cnt = 0
        start_frame = 0
        blank_cnt = 0
        last_100_time = time.time()
        while(True):
            ret, img = self.cap.read()
            frames_cnt+=1
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if (frames_cnt % 100 == 0 and frames_cnt != 0 and total_frames != 0):
                current_time = time.time()
                dtime = (current_time - last_100_time) / 100
                eta = dtime * (total_frames - frames_cnt)
                last_100_time = current_time
                print("Detected frame {} / {}, ETA: {} s".format(frames_cnt, total_frames, int(eta)))
            if frames_cnt < start_frame:
                continue
            if img is None:
                print("[WARN] Empty frame...")
                time.sleep(1)
                continue
            cv2.imshow("Raw", img)
            cv2.waitKey(1)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #检测黑白屏状态，以加速处理
            if np.mean(gray) < 10 and np.var(gray) < 10:
                blank_cnt+=1
                if blank_cnt % 10 == 0:
                    print("[WARN] Blank screen detected,Skipping this 10 frames")
                continue
            else:
                blank_cnt = 0
            # 进行推理,获取推理结果
            result = self.model.infer(img)
            # 若无结果继续进行下一帧识别
            if (len(result) == 0):
                continue

            labels = []
            # 遍历所有检测结果
            need_save = False
            vis_result(img, result)
            for det in result:
                target = get36ClsFrom64Cls(det)
                
                # 用于放行特定类型装甲板，以增加数据集中该类型装甲板比例，不需要可注释
                ##-----------Special Strategy Start---------------------
                id_str = "G12345OB"
                sz_str = "sb"
                id = id_str[det.id]
                size = sz_str[det.color % 2]
                
                if (not self.analyzer.isInDistribution(target.pts)) and (not need_save):
                    need_save = True
                if (id == "3" or id == "4" or id == "5") and size == "b":
                    need_save = True
                if id == "G":
                    need_save = True
                ##-----------Special Strategy End---------------------  
                
                label = str(int(target.cls))
                for pt in det.pts[:, :]:
                    coord_x = pt[0] / img.shape[1]
                    coord_y = pt[1] / img.shape[0]
                    label += " " + str(coord_x)
                    label += " " + str(coord_y)
                labels.append(label)
            cv2.waitKey(1)
            if (need_save):
                # 生成文件名并保存
                file_name = str(cnt).zfill(6)
                cv2.imwrite(os.path.join(self.dst_dir, file_name) + ".jpg", img)
                with open(os.path.join(self.dst_dir, file_name) + ".txt", "w+") as file:
                    for label in labels:
                        file.write(label + "\n")
                print("Saved data as {}".format(file_name))
                cnt += 1

    def autoLabelByImgs(self):
        # -----------------------初始化-----------------------------
        cnt = 0
        print("正在进行自动标注 ... ")
        # -----------------------初始化-----------------------------
        for img in tqdm(self.img_list):
            # 进行推理,获取推理结果
            result = self.model.infer(img)
            # dst =img.copy()
            # 若无结果继续进行下一帧识别
            if (len(result) == 0):
                continue
            labels = []
            # 遍历所有检测结果
            vis_result(img, result)
            for det in result:
                target = get36ClsFrom64Cls(det)
                
                label = str(int(target.cls))
                for pt in det.pts[:, :]:
                    coord_x = pt[0] / img.shape[1]
                    coord_y = pt[1] / img.shape[0]
                    label += " " + str(coord_x)
                    label += " " + str(coord_y)
                labels.append(label)
            file_name = str(cnt).zfill(6)
            cv2.imwrite(os.path.join(self.dst_dir, file_name) + ".jpg", img)
            with open(os.path.join(self.dst_dir, file_name) + ".txt", "w+") as file:
                for label in labels:
                    file.write(label + "\n")
            cnt += 1
        print("成功标注 {} 张图片 ...".format(cnt))
        print("="*50)

    def autoLabelByDaheng(self):
        """使用大恒相机自动标注
        """
        # -----------------------初始化-----------------------------
        cnt = 0
        # -----------------------初始化-----------------------------
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            os.sys.exit(1)
        strSN = dev_info_list[0].get("sn")
        # 通过序列号打开设备
        cam = device_manager.open_device_by_sn(strSN)
        # 开始采集
        cam.stream_on()
        cam.BalanceWhiteAuto.set(2)
        while (True):
            exposure_offset = random.randrange(-self.max_delta_exp, self.max_delta_exp)
            cam.ExposureTime.set(self.exp + exposure_offset)
            # 从第 0 个流通道获取一幅图像
            raw_image = cam.data_stream[0].get_image()
            # 从彩色原始图像获取 RGB 图像
            rgb_image = raw_image.convert("RGB")
            if rgb_image is None:
                continue
            # 从 RGB 图像数据创建 numpy 数组
            numpy_image = rgb_image.get_numpy_array()
            if numpy_image is None:
                continue

            img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            dst = copy(img)
            cv2.imshow("src", img)
            cv2.waitKey(1)
            # 进行推理,获取推理结果
            result = self.model.infer(img)
            # 若无结果继续进行下一帧识别
            if (len(result) == 0):
                continue

            labels = []
            # 遍历所有检测结果
            vis_result(img, result)
            need_save = False
            for det in result:
                target = get36ClsFrom64Cls(det)
                if (not self.analyzer.isInDistribution(target.pts)) and (not need_save):
                    need_save = True
                label = str(int(target.cls))
                for pt in det.pts[:, :]:
                    coord_x = pt[0] / img.shape[1]
                    coord_y = pt[1] / img.shape[0]
                    label += " " + str(coord_x)
                    label += " " + str(coord_y)
                labels.append(label)
            if (need_save):
                # 生成文件名并保存
                file_name = str(cnt).zfill(6)
                cv2.imwrite(os.path.join(self.dst_dir, file_name) + ".jpg", img)
                with open(os.path.join(self.dst_dir, file_name) + ".txt", "w+") as file:
                    for label in labels:
                        file.write(label + "\n")
                print("Saved data as {}".format(file_name))
                cnt += 1

    def run(self):
        print("="*50)
        print("数据集自动标注V1.2启动中")
        print("="*50)
        print("数据来源设置为  {} ...".format(self.type))
        if (self.type == "daheng"):
            self.using_daheng = True
        elif (self.type == "local_imgs"):
            self.using_local_imgs = True
            if os.path.exists(self.args.input_dir) is False:
                raise ValueError("无效的路径,请检查--input_dir参数")
            else:
                print("数据集目录设置为 {} ...".format(self.args.input_dir))
                img_list = getImgLists(self.args.input_dir, self.default_img_format)
                print("目标图像格式为 ", self.default_img_format)
                if len(img_list) == 0:
                    print("未检测到任何可用图片,程序将自动退出 ...")
                    sys.exit(1)
                else:
                    print("检测到 {} 张可用图片,即将开始自动标注 ...".format(len(img_list)))
        elif (self.type == "local_video"):
            self.using_local_video = True
            print("视频路径: {}".format(self.video_pth))
            self.cap = cv2.VideoCapture(self.video_pth)
        elif (self.type == "stream"):
            self.using_stream = True
            print("Address :{}".format(self.video_pth))
            real_pth = getLiveURL(self.video_pth)
            self.cap = cv2.VideoCapture(real_pth)
        else:
            raise ValueError("无效的输入类型,请检查--type参数")
        print("="*50)

        if (self.using_daheng):
            self.autoLabelByDaheng()
        elif (self.using_local_imgs):
            self.autoLabelByImgs()
        elif (self.using_local_video or self.using_stream):
            self.autoLabelByStream()
            

if __name__ == "__main__":
    annotator = Annotator()
    annotator.run()
