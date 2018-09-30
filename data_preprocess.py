
import numpy as np
import cv2


def optical_flow_2d(Train_X, Test_X, r, w, timesteps_TIM):
	
	Train_X_of = np.array([])

	for vid in np.arange(Train_X.shape[0]):

		of_array = np.array([])
		for step in np.arange(timesteps_TIM - 1):

			frame1 = Train_X[vid][:,step].reshape(r,w)
			frame2 = Train_X[vid][:,step+1].reshape(r,w)

			of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
			of = (of - np.min(of)) / (np.max(of) - np.min(of))
			of_array = np.append(of_array, of)

		Train_X_of = np.append(Train_X_of, of_array)

	Test_X_of = np.array([])
	for vid in np.arange(Test_X.shape[0]):
		
		of_array = np.array([])
		for step in np.arange(timesteps_TIM - 1):
			frame1 = Test_X[vid][:,step].reshape(r,w)
			frame2 = Test_X[vid][:,step+1].reshape(r,w)

			of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
			of = (of - np.min(of)) / (np.max(of) - np.min(of))
			of_array = np.append(of_array, of)

		Test_X_of = np.append(Test_X_of, of_array)

	Train_X_of = Train_X_of.reshape(Train_X.shape[0], 2, r, w).astype('float32')
	Test_X_of = Test_X_of.reshape(Test_X.shape[0], 2, r, w).astype('float32')

	return Train_X_of, Test_X_of
