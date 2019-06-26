import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from datetime import timedelta

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
	parser.add_argument('--coco_path', help='Path to COCO directory', default='coco')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--model', help='Path to model (.pt) file.', default='pretrained/coco_resnet_50_map_0_335.pt')
	parser.add_argument('--state_dict', help='Path to state dict (.pth) file.', default='pretrained/coco_resnet_50_map_0_335.pt')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	retinanet = torch.load(parser.model)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	video_list = ['camera_1.mp4','camera_2.mp4','camera_5.mp4','camera_6.mp4']
	for video in video_list:
		vidcap = cv2.VideoCapture(video)
		fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)	
		success,image = vidcap.read()
		count = 0
		total_time=0
		out = cv2.VideoWriter("final_vid/{}_detected.avi".format(video.split(".")[0]),cv2.cv.CV_FOURCC(*'XVID'), fps, (image.shape[1],image.shape[0]))

		with torch.no_grad():
			retinanet.eval()
			while success:
				image_2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

				image_2 = transform(image_2)
				image_2 = image_2.view((-1, image_2.size()[0],image_2.size()[1],image_2.size()[2]))
				st = time.time()
				scores, classification, transformed_anchors = retinanet(image_2.cuda().float())
				total_time += time.time()-st
				# print('Elapsed time: {}'.format(time.time()-st))
				idxs = np.where(scores>0.5)
				# print(torch.max(image_2),torch.min(image_2))
				# print(np.max(image),np.min(image))

				for j in range(idxs[0].shape[0]):
					bbox = transformed_anchors[idxs[0][j], :]
					x1 = int(bbox[0])
					y1 = int(bbox[1])
					x2 = int(bbox[2])
					y2 = int(bbox[3])
					label_name = dataset_val.labels[int(classification[idxs[0][j]])]
					draw_caption(image, (x1, y1, x2, y2), label_name)

					cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
					# print(label_name)

				# cv2.imwrite('final_vid/frame_{}.jpg'.format(count), image)
				
				out.write(image)
				success,image = vidcap.read()
				count+=1
				# if(count == 100):
				# 	break
			out.release()
		print("Total time taken is = {}".format(str(timedelta(seconds=total_time))))

if __name__ == '__main__':
 main()