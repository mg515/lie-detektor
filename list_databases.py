import numpy as np
import sys
import math
import operator
import csv
import glob,os
import xlrd
import pandas as pd

from utilities import Read_Input_Images, get_subfolders_num, data_loader_with_LOSO, label_matching, duplicate_channel
from utilities import record_scores, LossHistory, filter_objective_samples
from utilities import loading_samm_table, loading_smic_table, loading_casme_table,loading_casme1_table, ignore_casme_samples, ignore_casmergb_samples, loading_casme_objective_table
from utilities import get_vid_per_subject, get_vid_per_subject_augmented
#from samm_utilitis import get_subfolders_num_crossdb, loading_samm_labels


def load_db(db_path, list_db, spatial_size, objective_flag):
	db_name = list_db[0]
	db_home = db_path + db_name  + "/"
	db_images = db_path + db_name + "/" + db_name + "/"

	cross_db_flag = 0
	print("retrieving data for:" + db_name)


	if db_name == 'CASME2_TIM10':
		table = loading_casme_table(db_home + 'CASME2_label_Ver_3.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=26
		samples = 257
		n_exp = 4
		
		#VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		VidPerSubject,vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 1

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")

	elif db_name == 'CASME2_Optical' or db_name == 'CASME2_Optical2' or db_name == 'CASME2_Color_TIM10' or db_name == 'CASME2_Color_TIM20':
		print("arrived")
		table = loading_casme_table(db_home + 'CASME2_label_Ver_3.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=26
		if db_name != 'CASME2_Color_TIM10' and db_name != 'CASME2_Color_TIM20' and db_name != 'CASME2_Optical2':
			samples = 257
		else: samples = 255
		n_exp = 4

		#VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		#VidPerSubject = get_vid_per_subject(table, ['fear', 'sadness'])
		VidPerSubject,vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 3		

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")


	elif db_name == 'CASME1_Color_TIM20':
		print("arrived")
		table = loading_casme1_table(db_home + 'CASME1.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=19
		samples = 189
		n_exp = 4

		#VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		#VidPerSubject = get_vid_per_subject(table, ['fear', 'sadness'])
		VidPerSubject,vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 19
		data_dim = r * w
		channel = 3		

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")

	elif db_name == 'SMIC_TIM10':
		table = loading_smic_table(db_path, db_name)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		r = w = spatial_size
		subjects = 16
		samples = 164
		n_exp = 3
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		channel = 3

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")		

			

	elif db_name == 'SAMM_Optical':
		table, table_objective = loading_samm_table(db_path, db_name, objective_flag)
		# print(table)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 9
		data_dim = r * w
		channel = 3

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")		

	elif db_name == 'SAMM_TIM10':
		table, table_objective = loading_samm_table(db_path, db_name, objective_flag)
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		channel = 3
		#########################################################	

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")		

	elif db_name == 'SAMM_Strain':
		table, table_objective = loading_samm_table(db_path, db_name, objective_flag)

		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])

		################# Variables #############################
		r = w = spatial_size
		subjects = 29
		samples = 159
		n_exp = 8
		VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		timesteps_TIM = 10
		data_dim = r * w
		channel = 3
		#########################################################	

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")	


	elif db_name == 'SAMM_CASME_Optical':
		# total amount of videos 253
		table, table_objective = loading_samm_table(db_path, db_name)
		table = table_objective
		table2 = loading_casme_objective_table(db_path, db_name)

		# merge samm and casme tables
		table = np.concatenate((table, table2), axis=1)
		
		listOfIgnoredSamples = []
		IgnoredSamples_index = np.empty([0])
		sub_items = np.empty([0])
		list_samples = filter_objective_samples(table)

		r = w = spatial_size
		subjects = 47 # some subjects were removed because of objective classes and ignore samples: 47
		n_exp = 5
		samples = 253

		VidPerSubject, list_samples = get_subfolders_num_crossdb(db_images, IgnoredSamples_index, sub_items, table, list_samples)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 3

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")

		cross_db_flag = 1
		return r, w, subjects, samples, n_exp, VidPerSubject, timesteps_TIM, data_dim, channel, table, list_samples, db_home, db_images, cross_db_flag

	elif db_name == 'CASME2_Optical2_Aug' or db_name == 'CASME2_Optical_Aug' or db_name == 'CASME2_flowNet_Aug':
		print("arrived")
		table = loading_casme_table(db_home + 'CASME2_label_Ver_3.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=26
		samples = 200*4
		n_exp = 4

		#VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		VidPerSubject,vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 3		

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")

	elif db_name == 'CASME2_Cropped':
		print("arrived casme2 cropped")

		table = loading_casme_table(db_home + 'CASME2_label_Ver_3.xls', full = True)
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=26
		samples = 255
		n_exp = 4

		VidPerSubject, vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 1
		data_dim = r * w
		channel = 2

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")

	elif db_name == 'CASME12_Color_TIM10' or db_name == 'CASME12_flowNet':
		print("arrived")
		table = loading_casme1_table(db_home + 'CASME12.xls')
		listOfIgnoredSamples, IgnoredSamples_index = ignore_casme_samples(db_path, list_db)

		r = w = spatial_size
		subjects=45
		samples = 500*4
		n_exp = 4

		#VidPerSubject = get_subfolders_num(db_images, IgnoredSamples_index)
		VidPerSubject,vidList = get_vid_per_subject_augmented(db_images)

		timesteps_TIM = 9
		data_dim = r * w
		channel = 3		

		if os.path.exists(db_home + "Classification/" + db_name + "_label.txt" ) == True:
			os.remove(db_home + "Classification/" + db_name + "_label.txt")


	return r, w, subjects, samples, n_exp, VidPerSubject, vidList, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples, db_home, db_images, cross_db_flag

def restructure_data(subject, subperdb, labelpersub, subjects, n_exp, r, w, timesteps_TIM, channel):
	Train_X, Train_Y, Train_Y_gt, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(subject, subperdb, labelpersub, subjects, n_exp)
	# Rearrange Training labels into a vector of images, breaking sequence

	Train_X_spatial = Train_X.reshape(Train_X.shape[0]*timesteps_TIM, r, w, channel)
	Test_X_spatial = Test_X.reshape(Test_X.shape[0]* timesteps_TIM, r, w, channel)

	# Extend Y labels 10 fold, so that all images have labels
	Train_Y_spatial = np.repeat(Train_Y, timesteps_TIM, axis=0)
	Test_Y_spatial = np.repeat(Test_Y, timesteps_TIM, axis=0)	


	X = Train_X_spatial.reshape(Train_X_spatial.shape[0], r, w, channel)
	y = Train_Y_spatial.reshape(Train_Y_spatial.shape[0], n_exp)
	normalized_X = X.astype('float32') / 255.

	test_X = Test_X_spatial.reshape(Test_X_spatial.shape[0], r, w, channel)
	test_y = Test_Y_spatial.reshape(Test_Y_spatial.shape[0], n_exp)
	normalized_test_X = test_X.astype('float32') / 255.


	print ("Train_X_shape: " + str(np.shape(Train_X)))
	print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	print ("Test_X_shape: " + str(np.shape(Test_X)))
	print ("Test_Y_shape: " + str(np.shape(Test_Y)))	
	print ("X_shape: " + str(np.shape(X)))
	print ("y_shape: " + str(np.shape(y)))
	print ("test_X_shape: " + str(np.shape(test_X)))	
	print ("test_y_shape: " + str(np.shape(test_y)))	



	return Train_X, Train_Y, Test_Y, Test_Y, Test_Y_gt, X, y, test_X, test_y


def restructure_data_c3d(subject, subperdb, labelpersub, subjects, n_exp, r, w, timesteps_TIM, channel):

	Train_X, Train_Y, Train_Y_gt, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(subject, subperdb, labelpersub, subjects, n_exp)
	# Rearrange Training labels into a vector of images, breaking sequence

	#if timesteps_TIM == 1:
	#	Train_X = Train_X.reshape(Train_X.shape[0], channel, r, w).astype('float32') / 255.
#		Test_X = Test_X.reshape(Test_X.shape[0], channel, r, w).astype('float32') / 255.
#	else:
	Train_X = Train_X.reshape(Train_X.shape[0], channel, timesteps_TIM, r, w).astype('float32') / 255.
	Test_X = Test_X.reshape(Test_X.shape[0], channel, timesteps_TIM, r, w).astype('float32') / 255.

	print ("Train_X_shape: " + str(np.shape(Train_X)))
	print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	print ("Test_X_shape: " + str(np.shape(Test_X)))
	print ("Test_Y_shape: " + str(np.shape(Test_Y)))	
	#print ("X_shape: " + str(np.shape(X)))
	#print ("y_shape: " + str(np.shape(y)))
	#print ("test_X_shape: " + str(np.shape(test_X)))	
	#print ("test_y_shape: " + str(np.shape(test_y)))	



	return Train_X, Train_Y, Train_Y_gt, Test_X, Test_Y, Test_Y_gt



def restructure_data_apex(subject, subperdb, labelpersub, subjects, n_exp, r, w, timesteps_TIM, channel):
	
	Train_X, Train_Y, Train_Y_gt, Test_X, Test_Y, Test_Y_gt = data_loader_with_LOSO(subject, subperdb, labelpersub, subjects, n_exp)

	#size_of = 64

	# Rearrange Training labels into a vector of images, breaking sequence
	import cv2
	Train_X_of = np.array([])
	for vid in np.arange(Train_X.shape[0]):
		frame1 = Train_X[vid][0].reshape(r,w,3)
		frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		frame2 = Train_X[vid][1].reshape(r,w, 3)
		frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
		of = (of - np.min(of)) / (np.max(of) - np.min(of))

		#of_small = np.array([cv2.resize(of[:,:,0], (size_of,size_of)), cv2.resize(of[:,:,1], (size_of,size_of))])
		Train_X_of = np.append(Train_X_of, of)
	
	Test_X_of = np.array([])
	for vid in np.arange(Test_X.shape[0]):
		frame1 = Test_X[vid][0].reshape(r,w,3)
		frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
		frame2 = Test_X[vid][1].reshape(r,w, 3)
		frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
		of = (of - np.min(of)) / (np.max(of) - np.min(of))

		#of_small = np.array([cv2.resize(of[:,:,0], (size_of,size_of)), cv2.resize(of[:,:,1], (size_of,size_of))])
		Test_X_of = np.append(Test_X_of, of)
	
	Train_X = Train_X_of.reshape(Train_X.shape[0], channel, r, w).astype('float32')
	Test_X = Test_X_of.reshape(Test_X.shape[0], channel, r, w).astype('float32')


	print ("Train_X_shape: " + str(np.shape(Train_X)))
	print ("Train_Y_shape: " + str(np.shape(Train_Y)))
	print ("Test_X_shape: " + str(np.shape(Test_X)))
	print ("Test_Y_shape: " + str(np.shape(Test_Y)))	

	return Train_X, Train_Y, Train_Y_gt, Test_X, Test_Y, Test_Y_gt