
import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import cv2
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from reordering import readinput
from random import randint
from augmentation import *


def augment_crop(image, style, row=224, col=224, cutSize=4):
	style = randint(1,8)
	if style==1: return cv2.resize(image[cutSize:row,:], (col,row)) # cut from top
	elif style==2: return cv2.resize(image[0:(row-cutSize),:], (col,row)) # cut from bottom
	elif style==3: return cv2.resize(image[:,0:(col-cutSize)], (col,row)) # cut from right
	elif style==4: return cv2.resize(image[:,cutSize:col], (col,row)) # cut from left
	elif style==5: return cv2.resize(image[cutSize:row,cutSize:col], (col,row)) # topleft
	elif style==6: return cv2.resize(image[cutSize:row,0:(col-cutSize)], (col,row)) # topright
	elif style==7: return cv2.resize(image[0:(row-cutSize),0:(col-cutSize)], (col,row)) # bottomright
	elif style==8: return cv2.resize(image[0:(row-cutSize),cutSize:col], (col,row)) # bottomleft
	



def augment_image(img):

	if randint(0,1) == 1: img = augment_crop(img)
	elif randint(0,1) == 1: img,_ = flip(img)
	elif randint(0,1) == 1: img = rotation(randint(-8,8), img)

	return img


def augmentation_casme(db_images, outputDir, numSamples, table, resizedFlag, r, w):

	for emotion in ['positive', 'negative', 'surprise', 'others']:
		table_emotion = pd.DataFrame(data=table[0:,0:],columns=['sub','id','emotion'])
		table_emotion = table_emotion[table_emotion['emotion']==emotion]

		for i in range(numSamples):
			print(emotion+"_"+str(i))
			# first we ensure that every original video is processed, then we start sampling randomly until we have enough
			if i <= (table_emotion.shape[0]-1):
				random_pick = table_emotion.iloc[[i]] # not so random
				print("processing original video #" + str(i) + " emotion=" + emotion)
			else:
				random_pick = table_emotion.sample(n=1) # very random
				print("processing augmented video #" + str(i) + "emotion" + emotion)


			path = db_images+"sub"+str(random_pick['sub'].iloc[0])+"/"+str(random_pick['id'].iloc[0])+"/"

			imgList = readinput(path)
			numFrame = len(imgList)

			if resizedFlag == 1:
				col = w
				row = r
			else:
				img = cv2.imread(imgList[0])
				[row,col,_l] = img.shape

			for var in range(numFrame):
				img = cv2.imread(imgList[var])
				[_,_,dim] = img.shape

				if resizedFlag == 1:
					img = cv2.resize(img, (col,row))

				if i > (table_emotion.shape[0]-1):
					img = augment_image(img)

				writeFolder = outputDir+"sub"+str(random_pick['sub'].iloc[0])+"/"+str(random_pick['id'].iloc[0])+"."+str(i)+"/"
				outputPath = writeFolder + imgList[var].split('/')[-1]
				if not os.path.exists(writeFolder):
					os.makedirs(writeFolder)

				cv2.imwrite(outputPath, img)


