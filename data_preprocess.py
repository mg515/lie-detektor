
import numpy as np
import cv2


def optical_flow_2d(subperdb, samples, r, w, timesteps_TIM):


	subperdb_of = []

	for sub in subperdb:

		of_vids = []
		for vid in np.arange(len(sub)):

			of_array = []
			frame1 = sub[vid][0].reshape(r,w)
			for step in np.arange(timesteps_TIM):
				frame2 = sub[vid][step+1].reshape(r,w)

				of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
				of = (of - np.min(of)) / (np.max(of) - np.min(of))
				of_array.append(of)

			of_vids.append(of_array)
			
		subperdb_of.append(of_vids)

	return subperdb_of


def optical_flow_2d_old(Train_X, Test_X, r, w, timesteps_TIM):
	
	Train_X_of = np.array([])
	for vid in np.arange(Train_X.shape[0]):
 		frame1 = Train_X[vid][0,:].reshape(r,w)
 		frame2 = Train_X[vid][1,:].reshape(r,w)
 		of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
 		of = (of - np.min(of)) / (np.max(of) - np.min(of))
 		Train_X_of = np.append(Train_X_of, of)
	Test_X_of = np.array([])
	for vid in np.arange(Test_X.shape[0]):
 		frame1 = Test_X[vid][0,:].reshape(r,w)
 		frame2 = Test_X[vid][1,:].reshape(r,w)
 		of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)	
 		of = (of - np.min(of)) / (np.max(of) - np.min(of))
 		Test_X_of = np.append(Test_X_of, of)
		 
	Train_X_of = Train_X_of.reshape(Train_X.shape[0], 2, r, w).astype('float32')
	Test_X_of = Test_X_of.reshape(Test_X.shape[0], 2, r, w).astype('float32')
	return Train_X_of, Test_X_of
