import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
	path = './dataset/Labels/'
	files = os.listdir(path)
	all_bboxes = []
	for file in files:
		f = open('./dataset/Labels/'+file, 'r')
		bbox = [[file[:-4]+'.jpeg']+[int(j) for j in i[:-1].split()] for i in f.readlines()[1:]]
		f.close()
		_ = [all_bboxes.append(i) for i in bbox]
	all_bboxes = pd.DataFrame(np.array(all_bboxes))
	all_bboxes.columns = ['image_id', 'x1', 'y1', 'x2', 'y2']
	all_bboxes.to_csv('./dataset/labels.csv', index=False)