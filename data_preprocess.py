
import numpy as np
import cv2


def optical_flow_2d(subperdb, samples, r, w, timesteps_TIM, compareFrame1 = True):

	subperdb_of = []

	for sub in subperdb:

		of_vids = []
		for vid in np.arange(len(sub)):

			of_array = []
			if compareFrame1: frame1 = sub[vid][0].reshape(r,w)
			for step in np.arange(timesteps_TIM):
				if not compareFrame1:
					frame1 = sub[vid][step].reshape(r,w)
				frame2 = sub[vid][step+1].reshape(r,w)

				of = cv2.calcOpticalFlowFarneback(frame1,frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)		
				of = (of - np.min(of)) / (np.max(of) - np.min(of))
				of_array.append(of)

			of_vids.append(of_array)
			
		subperdb_of.append(of_vids)

	return subperdb_of
