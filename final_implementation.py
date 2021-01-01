import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
from dataset_loader import GunDataset

def collate_fn(batch):
	return tuple(zip(*batch))

class Averager:
	def __init__(self):
		self.current_total = 0.0
		self.iterations = 0.0

	def send(self, value):
		self.current_total += value
		self.iterations += 1

	@property
	def value(self):
		if self.iterations == 0:
			return 0
		else:
			return 1.0 * self.current_total / self.iterations

	def reset(self):
		self.current_total = 0.0
		self.iterations = 0.0

if __name__ == '__main__':

	DIR_INPUT = f'./dataset'
	DIR_TRAIN = f'{DIR_INPUT}/Images'

	train_df = pd.read_csv(f'{DIR_INPUT}/labels.csv')

	image_ids = train_df['image_id'].unique()
	valid_ids = image_ids[330:]
	train_ids = image_ids[:330]

	valid_df = train_df[train_df['image_id'].isin(valid_ids)]
	train_df = train_df[train_df['image_id'].isin(train_ids)]

	#########################################################################

	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	num_classes = 2  # 1 class (wheat) + background

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	#########################################################################

	train_dataset = GunDataset(train_df, DIR_TRAIN)
	valid_dataset = GunDataset(valid_df, DIR_TRAIN)


	# split the dataset in train and test set
	indices = torch.randperm(len(train_dataset)).tolist()
	'''
	train_data_loader = DataLoader(
		train_dataset,
		batch_size=16,
		shuffle=False,
		num_workers=4,
		collate_fn=collate_fn
	)
	'''
	valid_data_loader = DataLoader(
		valid_dataset,
		batch_size=8,
		shuffle=False,
		num_workers=4,
		collate_fn=collate_fn
	)

	device = torch.device('cpu')

	################################################################################

	model.to(device)
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
	# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
	lr_scheduler = None

	num_epochs = 2

	loss_hist = Averager()
	itr = 1

	for epoch in range(num_epochs):
		loss_hist.reset()
		
		for images, targets, image_ids in valid_data_loader:
			
			images = list(image.to(device) for image in images)
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

			loss_dict = model(images, targets)

			losses = sum(loss for loss in loss_dict.values())
			loss_value = losses.item()

			loss_hist.send(loss_value)

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			if itr % 1 == 0:
				print(f"Iteration #{itr} loss: {loss_value}")

			itr += 1
		
		# update the learning rate
		if lr_scheduler is not None:
			lr_scheduler.step()

		print(f"Epoch #{epoch} loss: {loss_hist.value}")

	images, targets, image_ids = next(iter(valid_data_loader))
	images = list(img.to(device) for img in images)
	targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
	boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)
	sample = images[1].permute(1,2,0).cpu().numpy()
	model.eval()
	cpu_device = torch.device("cpu")
	outputs = model(images)
	outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

	print(targets, outputs)