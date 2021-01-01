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
IMAGE_NO = 1
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
print("Valid Anchor Boxes Left:", valid_anchor_boxes.shape)