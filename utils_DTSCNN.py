
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

