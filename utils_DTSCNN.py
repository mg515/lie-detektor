
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



def Read_Input_Images_DTSCNN(inputDir, augmentationDir, dB, resizedFlag, table, workplace, r, w, channel):
	SubperdB = []

	# cross-checking parameter
	subperdb_id = []

	for sub in sorted([infile for infile in os.listdir(inputDir)]):
		VidperSub = [] 
		vid_id = np.empty([0])       

		for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
				

			import ipdb
			ipdb.set_trace()

            
			path = inputDir + sub + '/' + vid + '/' # image loading path
			if path in listOfIgnoredSamples:
				continue

			imgList = readinput(path)
			numFrame = len(imgList)

			if resizedFlag == 1:
				col = w
				row = r
			else:
				img = cv2.imread(imgList[0])
				[row,col,_l] = img.shape

			## read the label for each input video
			collectinglabel(table, sub[3:], vid, workplace+'Classification/', dB)


			for var in range(numFrame):
				img = cv2.imread(imgList[var])
					
				[_,_,dim] = img.shape
					
				if channel == 1:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				if resizedFlag == 1:
					img = cv2.resize(img, (col,row))
						
			
				if var == 0:
					FrameperVid = img.flatten()
				else:
					FrameperVid = np.vstack((FrameperVid,img.flatten()))
					
				vid_id = np.append(vid_id, imgList[var]) # <--cross-check
			VidperSub.append(FrameperVid)       
	
		subperdb_id.append(vid_id)# <--cross-check
		SubperdB.append(VidperSub)	

	return SubperdB






def Read_Input_Images_DTSCNN(inputDir, listOfIgnoredSamples, dB, resizedFlag, table, workplace, r, w, channel):
	SubperdB = []

	# cross-checking parameter
	subperdb_id = []

	for sub in sorted([infile for infile in os.listdir(inputDir)]):
		VidperSub = [] 
		vid_id = np.empty([0])       

		for vid in sorted([inrfile for inrfile in os.listdir(inputDir+sub)]):
            
			path = inputDir + sub + '/' + vid + '/' # image loading path
			if path in listOfIgnoredSamples:
				continue

			imgList = readinput(path)
			numFrame = len(imgList)

			if resizedFlag == 1:
				col = w
				row = r
			else:
				img = cv2.imread(imgList[0])
				[row,col,_l] = img.shape

			## read the label for each input video
			collectinglabel(table, sub[3:], vid, workplace+'Classification/', dB)


			for var in range(numFrame):
				img = cv2.imread(imgList[var])
					
				[_,_,dim] = img.shape
					
				if channel == 1:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				if resizedFlag == 1:
					img = cv2.resize(img, (col,row))
						
			
				if var == 0:
					FrameperVid = img.flatten()
				else:
					FrameperVid = np.vstack((FrameperVid,img.flatten()))
					
				vid_id = np.append(vid_id, imgList[var]) # <--cross-check
			VidperSub.append(FrameperVid)       
	
		subperdb_id.append(vid_id)# <--cross-check
		SubperdB.append(VidperSub)	

	return SubperdB



def augment_image(image, style, row, col, cutSize):

	if style==0:
		return image
	elif style==1: return cv2.resize(image[cutSize:row,:], (col,row)) # cut from top
	elif style==2: return cv2.resize(image[0:(row-cutSize),:], (col,row)) # cut from bottom
	elif style==3: return cv2.resize(image[:,0:(col-cutSize)], (col,row)) # cut from right
	elif style==4: return cv2.resize(image[:,cutSize:col], (col,row)) # cut from left
	elif style==5: return cv2.resize(image[cutSize:row,cutSize:col], (col,row)) # topleft
	elif style==6: return cv2.resize(image[cutSize:row,0:(col-cutSize)], (col,row)) # topright
	elif style==7: return cv2.resize(image[0:(row-cutSize),0:(col-cutSize)], (col,row)) # bottomright
	elif style==8: return cv2.resize(image[0:(row-cutSize),cutSize:col], (col,row)) # bottomleft
	






def augmentation_casme(db_images, outputDir, table, resizedFlag, r, w):

	for emotion in ['positive', 'negative', 'surprise', 'others']:
		table_emotion = pd.DataFrame(data=table[0:,0:],columns=['sub','id','emotion'])
		table_emotion = table_emotion[table_emotion['emotion']==emotion]

		for i in range(500):
			print(emotion+"_"+str(i))
			random_pick = table_emotion.sample(n=1)
			path = db_images+"sub"+str(random_pick['sub'].iloc[0])+"/"+str(random_pick['id'].iloc[0])+"/"

			imgList = readinput(path)
			numFrame = len(imgList)

			if resizedFlag == 1:
				col = w
				row = r
			else:
				img = cv2.imread(imgList[0])
				[row,col,_l] = img.shape

			### read the label for each input video
			#collectinglabel(table, sub[3:], vid, workplace+'Classification/', dB)


			for var in range(numFrame):
				img = cv2.imread(imgList[var])
					
				[_,_,dim] = img.shape
					
				#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

				if resizedFlag == 1:
					img = cv2.resize(img, (col,row))

				img = augment_image(img, randint(0,8), row, col, 4)

				writeFolder = outputDir+"sub"+str(random_pick['sub'].iloc[0])+"/"+str(random_pick['id'].iloc[0])+"."+str(i)+"/"
				outputPath = writeFolder + imgList[var].split('/')[-1]
				if not os.path.exists(writeFolder):
					os.makedirs(writeFolder)

				cv2.imwrite(outputPath, img)


