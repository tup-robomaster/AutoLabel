import os
import cv2
from cv2 import normalize
import numpy as np
from sympy import false
from tqdm import tqdm
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import yaml

class DistributionAnalyzer():
    def __init__(self, cfg="dataset_cfg.yaml"):
        self.cls_data = []
        self.pts_data = []
        self.pts_data_normed = []
        self.pts_distibution_heatmap = []
        self.label_file_map = []
        self.maxs = []
        self.mins = []
        self.grids = []
        
        self.grid_scale = 0
        self.grid_num = 0
        self.num_pts = 0
        self.num_cls = 0
        self.num_colors = 0
        self.OOD_thres = 0
        
        #Get cfg from Raw Data.
        with open(cfg, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.load(f.read(), yaml.FullLoader)
            self.dir =  yaml_dict["dataset_path"]
            self.num_pts = yaml_dict["num_pts"]
            self.num_cls = yaml_dict["num_cls"]
            self.grid_num = yaml_dict["grid_num"]
            self.OOD_thres = yaml_dict["OOD_thres"]
            self.pts_distibution_heatmap = np.zeros([self.num_pts - 2, self.grid_num, self.grid_num])
            
        #Get raw dicts
        print("Reading prior labels...")
        for root, dirs, files in os.walk(self.dir, topdown=False):
            for file_name in tqdm(files):
                with open(os.path.join(root,file_name),'r+') as src:
                    lines = src.readlines()
                    for line in lines:
                        label = []
                        text = line.strip().split()
                        cls = int(float(text[0]))
                        pts = []
                        for i in range(self.num_pts):
                            pts.append([float(text[1 + 2 * i]),float(text[2 * (i+1)])])
                        self.cls_data.append(cls)
                        self.pts_data.append(pts)
                        self.label_file_map.append(file_name)
        self.cls_data = np.array(self.cls_data)
        self.pts_data = np.array(self.pts_data)

        self.getPtsDistribution()
        fig = plt.figure()
        for i in range(self.num_pts - 2):
            plt.subplot(1, self.num_pts - 2, i+1)
            plt.title("Pt {}".format(i+3))
            plt.imshow(self.pts_distibution_heatmap[i])
            plt.colorbar()
        print("Showing distribution result...")
        plt.show()
    
    def getPtsDistribution(self):
        for _, label_pts in enumerate(self.pts_data):
            transformed_label = self.normalize(label_pts)
            self.pts_data_normed.append(transformed_label)
        
        self.getGrids()
        #Walk all normalized label to calcuate frequency.
        cnt = 0
        for pts_normed in self.pts_data_normed:
            data = self.getHeatMapCoord(pts_normed)
            data = np.floor(data).astype(np.int32)
            for i in range(self.num_pts - 2):
                self.pts_distibution_heatmap[i,data[i],data[i+1]]+=1
            cnt+=1
        self.pts_distibution_heatmap/=len(self.cls_data)
        
    def normalize(self, pts):
        affine_mat, _= cv2.estimateAffinePartial2D(pts[0:2], np.array([[0,-1],[0,1]]))
        affine_homomat = np.vstack((affine_mat,np.array([0,0,1])))
        transformed_label = np.zeros([self.num_pts,2])
        for i in range(self.num_pts):
            xy_from = pts[i].reshape(-1,1)
            transformed_label[i] = np.matmul(affine_homomat,np.vstack((xy_from,[1])))[:-1].T
        # print(transformed_label)
        return transformed_label
    
    def getGrids(self):
        #Walk all normalized label to get self.maxs and self.mins.
        self.maxs = np.zeros(2 * (self.num_pts - 2))
        self.mins = np.zeros(2 * (self.num_pts - 2))
        for pts_normed in self.pts_data_normed:
            for i in range(self.num_pts - 2):
                if pts_normed[i+2][0] > self.maxs[2 * i]:
                    self.maxs[2 * i] = pts_normed[i+2][0]
                elif pts_normed[i+2][0] < self.mins[2 * i]:
                    self.mins[2 * i] = pts_normed[i+2][0]

                if pts_normed[i+2][1] > self.maxs[2 * i + 1]:
                    self.maxs[2 * i + 1] = pts_normed[i + 2][1]
                elif pts_normed[i+2][1] < self.mins[2 * i + 1]:
                    self.mins[2 * i + 1] = pts_normed[i + 2][1]
        self.mins -=1
        self.grids = (self.maxs - self.mins) / self.grid_num
    
    def getHeatMapCoord(self, pts):
        return (self.maxs - pts[2:].reshape(1,-1)[0]) / self.grids
    
    def isInDistribution(self, pts,vis=True):
        pts = self.normalize(pts)
        data = self.getHeatMapCoord(pts)
        data = np.floor(data).astype(np.int32)
        if vis:
            img = np.array(self.pts_distibution_heatmap)
            pt_scalar = (0,0,0)
            is_ood_exists = False
            for i in range(self.num_pts - 2):
                if self.pts_distibution_heatmap[i,data[i],data[i+1]] <= self.OOD_thres:
                    pt_scalar = (0,0,255)
                    is_ood_exists = True
                else:
                    pt_scalar = (0,255,0)
                img_raw = img[i]
                normalized_data = cv2.normalize(img_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 使用 applyColorMap 函数生成三维彩色热力图
                heatmap = cv2.applyColorMap(normalized_data, cv2.COLORMAP_JET)
                cv2.circle(heatmap,(data[i+1],data[i]),3,pt_scalar,1)
                heatmap_resized = cv2.resize(heatmap,(512,512))
                # 显示热力图
                cv2.namedWindow('Heatmap@Pt {}', cv2.WINDOW_NORMAL)
                cv2.imshow('Heatmap@Pt {}'.format(i+3), heatmap_resized)
                cv2.waitKey(1)
            return not is_ood_exists
        else:
            for i in range(self.num_pts - 2):
                if self.pts_distibution_heatmap[i][data[i]][data[i+1]] <= self.OOD_thres:
                    return False
            return True