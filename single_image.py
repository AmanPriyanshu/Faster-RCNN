import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

SHOW_IMAGES = False
IMAGE_NO = 2
IMAGE_NO = str(IMAGE_NO)

### Enabling CUDA

if(torch.cuda.is_available()):
	device = torch.device("cuda")
	print(device, torch.cuda.get_device_name(0))
else:
	device= torch.device("cpu")
	print(device)

### Experimenting with a single image:

# input image could be of any size
img_og = cv2.imread('./dataset/Images/'+IMAGE_NO+'.jpeg')
img_og = cv2.cvtColor(img_og, cv2.COLOR_BGR2RGB) 
print("Original Image Dimesnions:", img_og.shape)
plt.imshow(img_og)
if SHOW_IMAGES:
	plt.show()

# input for bbox
file = open('./dataset/Labels/'+IMAGE_NO+'.txt', 'r')
bbox_og = np.array([[int(j) for j in i[:-1].split()] for i in file.readlines()[1:]])
file.close()

print("BBOX: ", bbox_og)

### Generating images with Bounding Boxes:

img_clone = np.copy(img_og)
for i in range(len(bbox_og)):
	cv2.rectangle(img_clone, (bbox_og[i][0], bbox_og[i][1]), (bbox_og[i][2], bbox_og[i][3]), color=(0, 255, 0), thickness=3)  
plt.imshow(img_clone)
if SHOW_IMAGES:
	plt.show()

### Resizing Image and BBOX:

img = cv2.resize(img_og, dsize=(800, 800), interpolation=cv2.INTER_CUBIC)
print("Resized Image Dimesnions:", img.shape)
plt.imshow(img)
if SHOW_IMAGES:
	plt.show()

# change the bounding box coordinates 
width_ratio = 800/img_og.shape[1]
height_ratio = 800/img_og.shape[0]
ratios = [width_ratio, height_ratio, width_ratio, height_ratio]
bbox = []
for box in bbox_og:
	box = [int(a * b) for a, b in zip(box, ratios)] 
	bbox.append(box)
bbox = np.array(bbox)
print("RESIZED BBOX:",bbox)

labels = [1]*len(bbox)  # Since we have only one label, i.e. Gun all Labels are 1

# display bounding box and labels
img_clone = np.copy(img)
bbox_clone = bbox.astype(int)
for i in range(len(bbox)):
	cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color=(0, 255, 0), thickness=3) # Draw Rectangle
plt.imshow(img_clone)
if SHOW_IMAGES:
	plt.show() 

### VGG16 preparation:

# Loading VGG16
model = torchvision.models.vgg16(pretrained=True).to(device)
fe = list(model.features)
print("\n\nVGG16 features:", len(fe), fe, '\n\n')

# collect layers with output feature map size (W, H) > limit=50
limit = 50
dummy_img = torch.zeros((1, 3, 800, 800)).float() # test image array [1, 3, 800, 800] 
print(dummy_img.shape)

req_features = []
k = dummy_img.clone().to(device)
for i in fe:
	k = i(k)
	print(k.size())
	if k.size()[2] < limit:
		break
	req_features.append(i)
	out_channels = k.size()[1]
print(len(req_features)) #30
print(out_channels, '\n\n') # 512

# Now converting this model into a feature_extractor:
faster_rcnn_feture_extractor = nn.Sequential(*req_features)

# Defining a simple Torch Transformer:
transform = transforms.Compose([transforms.ToTensor()])

imgTensor = transform(img).to(device) 
imgTensor = imgTensor.unsqueeze(0)
out_map = faster_rcnn_feture_extractor(imgTensor)
print("Output Size after Feature Extraction:", out_map.size(), '\n\n')

### Generating Anchor Boxes:

fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)

# coordinates of the 2500 center points to generate anchor boxes
index = 0
ctr = np.zeros((2500, 2))
for x in range(len(ctr_x)):
	for y in range(len(ctr_y)):
		ctr[index, 1] = ctr_x[x] - 8
		ctr[index, 0] = ctr_y[y] - 8
		index +=1
print("Anchor Box Centres:", ctr.shape)

# display the 2500 anchors
img_clone = np.copy(img)
plt.figure(figsize=(9, 6))
for i in range(ctr.shape[0]):
	cv2.circle(img_clone, (int(ctr[i][0]), int(ctr[i][1])), radius=1, color=(255, 0, 0), thickness=1) 
plt.imshow(img_clone)
if SHOW_IMAGES:
	plt.show()

# for each of the 2500 anchors, generate 9 anchor boxes
# 2500*9 = 22500 anchor boxes. Now each anchor box is generated using scales and ratios, 
# where, scale would represent size scale increase/decrease. Ratio on the other hand 
# would differentiate between images of the same scale. The height and width are 
# inversely related.

ratios = [0.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 16
anchor_boxes = np.zeros( ((fe_size * fe_size * 9), 4))
index = 0
for c in ctr:
	ctr_y, ctr_x = c
	for i in range(len(ratios)):
		for j in range(len(scales)):
			h = sub_sample * scales[j] * np.sqrt(ratios[i])
			w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
			anchor_boxes[index, 0] = ctr_x - w / 2.
			anchor_boxes[index, 1] = ctr_y - h / 2.
			anchor_boxes[index, 2] = ctr_x + w / 2.
			anchor_boxes[index, 3] = ctr_y + h / 2.
			index += 1
print("Anchor Boxes", anchor_boxes.shape)

# display the 9 anchor boxes of one anchor and the ground trugh bbox
img_clone = np.copy(img)
for i in range(11025, 11034):  #9*1225=11025
	x0 = int(anchor_boxes[i][0])
	y0 = int(anchor_boxes[i][1])
	x1 = int(anchor_boxes[i][2])
	y1 = int(anchor_boxes[i][3])
	cv2.rectangle(img_clone, (x0, y0), (x1, y1), color=(255, 255, 255), thickness=3) 

for i in range(len(bbox)):
	cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color=(0, 255, 0), thickness=3) # Draw Rectangle
	
plt.imshow(img_clone)
if SHOW_IMAGES:
	plt.show() 

# Ignore out-of-boundary anchor boxes
# valid anchor boxes with (y1, x1)>0 and (y2, x2)<=800
index_inside = np.where(
		(anchor_boxes[:, 0] >= 0) &
		(anchor_boxes[:, 1] >= 0) &
		(anchor_boxes[:, 2] <= 800) &
		(anchor_boxes[:, 3] <= 800)
	)[0]
print("Number of Boxes with Boundary:", index_inside.shape)

valid_anchor_boxes = anchor_boxes[index_inside]
print("Valid Anchor Boxes Left:", valid_anchor_boxes.shape, "\n\n")

### Calculate iou of the valid anchor boxes 

# IOU = intersection(A, Y)/Union(A, Y)
# Since we have 8940 anchor boxes and 1 ground truth object, we should get an array with (8490, 1) as the output. 
ious = np.empty((len(valid_anchor_boxes), len(bbox)), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(valid_anchor_boxes):
	xa1, ya1, xa2, ya2 = i  
	anchor_area = (ya2 - ya1) * (xa2 - xa1)
	for num2, j in enumerate(bbox):
		yb1, xb1, yb2, xb2 = j
		box_area = (yb2 - yb1) * (xb2 - xb1)
		inter_x1 = max([xb1, xa1])
		inter_y1 = max([yb1, ya1])
		inter_x2 = min([xb2, xa2])
		inter_y2 = min([yb2, ya2])
		if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
			iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
			iou = iter_area / (anchor_area + box_area - iter_area)			
		else:
			iou = 0.
		ious[num1, num2] = iou
print("IOU of generated Anchor Boxes:", ious.shape)

# What anchor box has max iou with the ground truth bbox  
gt_argmax_ious = ious.argmax(axis=0)
print("Index of Maximum IOU with ground truth box:", gt_argmax_ious)

gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
print("Maximum IOU with ground truth box:", gt_max_ious)

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print("Other indexes with same IOU:", gt_argmax_ious)

# What ground truth bbox is associated with each anchor box 
argmax_ious = ious.argmax(axis=1)
print("Ground Truth Box Associated with each Anchor Box - shape:", argmax_ious.shape)
print("Ground Truth Box Associated with each Anchor Box - values:",argmax_ious)
max_ious = ious[np.arange(len(index_inside)), argmax_ious]
print("Max IOUs:", max_ious, "\n\n")

### Labelling:
label = np.ones((len(index_inside), ), dtype=np.int32) * -1
print("Labels Shape", label.shape)

# Use iou to assign 1 (objects) to two kind of anchors 
# a) The anchors with the highest iou overlap with a ground-truth-box
# b) An anchor that has an IoU overlap higher than 0.6 with ground-truth box

# Assign 0 (background) to an anchor if its IoU ratio is lower than 0.3 for all ground-truth boxes
pos_iou_threshold  = 0.6
neg_iou_threshold = 0.3
label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1
label[max_ious < neg_iou_threshold] = 0

print("After Labelling:", label, "\n\n")

### Batches - Ignoring: -1, Positive: 1, Negative: 0

n_sample = 256
pos_ratio = 0.5
n_pos = pos_ratio * n_sample

pos_index = np.where(label == 1)[0]
if len(pos_index) > n_pos:
	disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
	label[disable_index] = -1
	
n_neg = n_sample * np.sum(label == 1)
neg_index = np.where(label == 0)[0]
if len(neg_index) > n_neg:
	disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
	label[disable_index] = -1

print("1s chosen:",np.where(label==1)[0].shape[0], "| 0s chosen:", np.where(label==0)[0].shape[0], "| Ignored Chosen:", np.where(label==-1)[0].shape[0], "\n\n")

### LOCS
# For each valid anchor box, find the groundtruth object which has max_iou 
max_iou_bbox = bbox[argmax_ious]
print("Max IOU BBOX:", max_iou_bbox.shape)

# valid anchor boxes h, w, cx, cy 
width = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
height = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 1] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 0] + 0.5 * width

# valid anchor box max iou bbox h, w, cx, cy 
base_width = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_height = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 1] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 0] + 0.5 * base_width

# valid anchor boxes  loc = (y-ya/ha), (x-xa/wa), log(h/ha), log(w/wa)
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps) # height !=0
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print("Anchor Locs:",anchor_locs.shape)

anchor_labels = np.empty((len(anchor_boxes),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label
print("Anchor Labels:", anchor_labels.shape)

anchor_locations = np.empty((len(anchor_boxes),) + anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs
print("Anchor Locations:", anchor_locations.shape, "\n\n")

### RPN and ROI:

in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
mid_channels = 512
n_anchor = 9  # Number of anchors at each location

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1).to(device)
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

reg_layer = nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0).to(device)
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

cls_layer = nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0).to(device) ## I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

x = conv1(out_map.to(device)) # out_map = faster_rcnn_fe_extractor(imgTensor)
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

# RPN anchor box format 
# [1, 36(9*4), 50, 50] => [1, 22500(50*50*9), 4] (dy, dx, dh, dw)
# [1, 18(9*2), 50, 50] => [1, 22500, 2]  (1, 0)
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
print("Locs Predction:", pred_anchor_locs.shape)

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
print("Classification Scores Prediction:", pred_cls_scores.shape)

objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
print("Objectness Format:", objectness_score.shape)

pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
print("Readable Format:", pred_cls_scores.shape)

### ROI and RPN Loss:

