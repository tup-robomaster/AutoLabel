
# -*- coding: utf-8 -*-

from copy import copy
from sqlite3 import Time
import string
from attr import attr
import cv2
import argparse
from model_64cls import Model
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

analyzer = DistributionAnalyzer()

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


def autoLabelByStream(model, cap):
    """使用视频流自动标注
    Args:
        model (Model): Model类
        cap (VideoCapture): 视频流
    """
    # -----------------------初始化-----------------------------
    cnt = 0
    frames_cnt = 0
    start_frame = 0
    last_100_time = time.time()
    path = "SJTU/"
    while(True):
        ret, img = cap.read()
        frames_cnt+=1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        if np.mean(img) < 0.1:
            print("[WARN] Black screen detected,Skipping this frame")
            continue
        # 进行推理,获取推理结果
        result = model.infer(img)
        # 若无结果继续进行下一帧识别
        if (len(result) == 0):
            continue

        labels = []
        # 遍历所有检测结果
        need_save = False
        vis_result(img, result)
        for det in result:
            target = get36ClsFrom64Cls(det)
            
            id_str = "G12345OB"
            sz_str = "sb"
            id = id_str[det.id]
            size = sz_str[det.color % 2]
            
            if (not analyzer.isInDistribution(target.pts)) and (not need_save):
                need_save = True
            if (id == "3" or id == "4" or id == "5") and size == "b":
                need_save = True
            if id == "G":
                need_save = True
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
            cv2.imwrite(os.path.join(path, file_name) + ".jpg", img)
            with open(os.path.join(path, file_name) + ".txt", "w+") as file:
                for label in labels:
                    file.write(label + "\n")
            print("Saved data as {}".format(file_name))
            cnt += 1


def autoLabelByImgs(model, img_list):
    # -----------------------初始化-----------------------------
    cnt = 0
    path = "Data/"
    print("正在进行自动标注 ... ")
    # -----------------------初始化-----------------------------
    for img in tqdm(img_list):
        # 进行推理,获取推理结果
        result = model.infer(img)
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
        cv2.imwrite(os.path.join(path, file_name) + ".jpg", img)
        with open(os.path.join(path, file_name) + ".txt", "w+") as file:
            for label in labels:
                file.write(label + "\n")
        cnt += 1
    print("成功标注 {} 张图片 ...".format(cnt))
    print("="*50)

def autoLabelByDaheng(model, exp, max_delta_exp):
    """使用大恒相机自动标注
    Args:
        model (Model): Model类
        exp (int): 平均曝光
        max_delta_exp (int): 最大曝光差值
    """
    # -----------------------初始化-----------------------------
    cnt = 0
    path = "Data/"
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
        exposure_offset = random.randrange(-max_delta_exp, max_delta_exp)
        cam.ExposureTime.set(exp + exposure_offset)
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
        result = model.infer(img)
        # 若无结果继续进行下一帧识别
        if (len(result) == 0):
            continue

        labels = []
        # 遍历所有检测结果
        need_save = False
        vis_result(img, result)
        for det in result:
            target = get36ClsFrom64Cls(det)
            print(det)
            if (not analyzer.isInDistribution(target.pts)) and (not need_save):
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
            cv2.imwrite(os.path.join(path, file_name) + ".jpg", img)
            with open(os.path.join(path, file_name) + ".txt", "w+") as file:
                for label in labels:
                    file.write(label + "\n")
            print("Saved data as {}".format(file_name))
            cnt += 1

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

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--type", "-t", type=str, help="输入类型,支持daheng,local_imgs, local_video, stream.", default="local_video")
    parser.add_argument("--input_dir", "-i", help="目标数据位置.")
    parser.add_argument("--video", help="目标视频位置.", default="/home/rangeronmars/Videos/Sentry/2022-06-25_16_13_43.avi")
    args = parser.parse_args()
    # --------------------------初始化变量-----------------------------
    default_img_format = [".jpg", ".png", ".jpeg"]
    # model = Model("model-opt.onnx")
    model = Model("model_64cls.onnx")
    mean_exposure = 7500
    max_delta_exposure = 2500
    using_daheng = False
    using_local_imgs = False
    using_local_video = False
    using_stream = False
    type = args.type
    img_lists = []
    video_pth = ""
    # --------------------------初始化完成-----------------------------
    # 输入类型设置
    print("="*50)
    print("数据集自动标注V1.2启动中")
    print("="*50)
    print("数据来源设置为  {} ...".format(type))
    if (type == "daheng"):
        using_daheng = True
    elif (type == "local_imgs"):
        using_local_imgs = True
        if os.path.exists(args.input_dir) is False:
            raise ValueError("无效的路径,请检查--input_dir参数")
        else:
            print("数据集目录设置为 {} ...".format(args.input_dir))
            img_list = getImgLists(args.input_dir, default_img_format)
            print("目标图像格式为 ", default_img_format)
            if len(img_list) == 0:
                print("未检测到任何可用图片,程序将自动退出 ...")
                sys.exit(1)
            else:
                print("检测到 {} 张可用图片,即将开始自动标注 ...".format(len(img_list)))
    elif (type == "local_video"):
        using_local_video = True
        video_pth = args.video
        print("视频路径: {}".format(video_pth))
        cap = cv2.VideoCapture(video_pth)
    elif (type == "stream"):
        using_stream = True
        video_pth = args.video
        print("Address :{}".format(video_pth))
        real_pth = getLiveURL(video_pth)
        cap = cv2.VideoCapture(real_pth)
    else:
        raise ValueError("无效的输入类型,请检查--type参数")
    print("="*50)

    if (using_daheng):
        autoLabelByDaheng(model, mean_exposure, max_delta_exposure)
    elif (using_local_imgs):
        autoLabelByImgs(model, img_list)
    elif (using_local_video or using_stream):
        autoLabelByStream(model, cap)


if __name__ == "__main__":
    main()
