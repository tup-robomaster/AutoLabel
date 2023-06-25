"""
使用CNN计算图像间的相似度,进行图像清洗.
--root_dir 输入根路径
--save_path 输出根路径
--start_num 起始计数.
"""
import argparse
from logging import root
import cv2
from imagededup.methods import CNN
from imagededup.utils import plot_duplicates
import os
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="label",type=str, help="输入根路径.")
parser.add_argument("--save_path", type=str,default="output", help="输出根路径.")
parser.add_argument("--sim_thres", type=float,default=0.99, help="相似度阈值.")
arg = parser.parse_args()

def GenerateFileName(index,extend_name):
	file_name = str(index).zfill(4)
	file_name_img = file_name + extend_name
	file_name_lab = file_name + ".txt"
	return file_name_img, file_name_lab

def main():
	default_img_format = [".jpg",".png",".jpeg"]
	img_list = []
	blur_list = []
	phasher = CNN()
	src_dir = arg.root_dir
	out_dir = arg.save_path
	thres = arg.sim_thres

	#Generate img list to travel
	for root, dirs, files in os.walk(src_dir, topdown=False):
		for file_name in files:
			# Continue while the file is illegal.
			for extend in default_img_format:
				if file_name.endswith(extend):
					img_list.append(file_name)
					break
	# for img_file in img_list:
	# 	img = cv2.imread(os.path.join(src_dir, img_file))
	# 	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 	mask = gray > gray.mean()
	# 	lap = cv2.Laplacian(gray,cv2.CV_64F)
	# 	score = lap[mask].var()
	# 	print("mean:{}".format(gray.mean()))
	# 	print(score)
	# 	print("="*50)
	# 	cv2.imshow("img",img)
	# 	cv2.waitKey(2000)
	#Generateing CNN encoding
	print('='*20)
	print("Image list generated,generating CNN encodings...")
	print('='*20)
	encodings = phasher.encode_images(src_dir)
	# duplicates_to_show = phasher.find_duplicates(encoding_map=encodings, min_similarity_threshold=thres)
	duplicates = phasher.find_duplicates_to_remove(encoding_map=encodings, min_similarity_threshold=thres)
	duplicates.sort()
	# print(duplicates)

	# Generating Final List
	print('='*20)
	print("Find duplicates done, generating Final list...")
	print('='*20)
	img_to_copy = img_list.copy()
	# for img_to_remove in duplicates_to_show:
		# if (len(duplicates_to_show[img_to_remove]) != 0):
		# 	plot_duplicates(image_dir=arg.root_dir, duplicate_map=duplicates_to_show, filename=img_to_remove)

	# print("Find duplicates done,generating Final list...")
	# print('='*20)
	img_to_copy = img_list.copy()
	for img_to_remove in duplicates:
		img_to_copy.remove(img_to_remove)

	#Copying accepted images
	print('='*20)
	print("Final list generated,Copying accepted images...")
	print('='*20)
	for img_name in img_to_copy:
		copyfile(os.path.join(src_dir, img_name), os.path.join(out_dir, img_name))

	print('='*20)
	print("Done...")
	print("Accepted {} images successfully.".format(str(len(img_to_copy))))
	print('='*20)

if __name__ == "__main__":
	main()
