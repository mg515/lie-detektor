

#path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Color_TIM10/CASME2_Color_TIM10/'
#pathtxt = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Color_TIM10/'


db_name = 'CASME1_Color_TIM10'

path = '/home/mihag/workingDir/flownet2-docker/data/' + db_name + '/' + db_name + '/'
pathtxt = '/home/mihag/workingDir/flownet2-docker/data/' + db_name + '/'



import os	
file1 = open(pathtxt + 'clip1.txt','w') 
file2 = open(pathtxt + 'clip2.txt','w') 
file3 = open(pathtxt + 'clip_flow.txt','w')

folders_sub = sorted(os.listdir(path))
for sub in folders_sub:
	print(sub)
	folders_clip = sorted(os.listdir(path + sub))
	for clip in folders_clip:
		print(clip)
		pics = sorted(os.listdir(path + sub + '/' + clip))

		for pic in pics[:-1]:
			file1.write('data/' + db_name + '/' + db_name + '/' + sub + '/' + clip + '/' + pic + '\n')
			pic = pic.split('.')[0] + '.flo'
			file3.write('data/' + db_name + '/' + db_name + '/' + sub + '/' + clip + '/' + pic + '\n')
		for pic in pics[1:]:
			file2.write('data/' + db_name + '/' + db_name + '/' + sub + '/' + clip + '/' + pic + '\n')

file1.close()
file2.close()
file3.close()
	
#	return True
