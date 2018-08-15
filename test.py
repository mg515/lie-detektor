

#path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Color_TIM10/CASME2_Color_TIM10/'
#pathtxt = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Color_TIM10/'


path = '/home/mihag/workingDir/flownet2-docker/data/CASME2_Color_TIM10/CASME2_Color_TIM10/'
pathtxt = '/home/mihag/workingDir/flownet2-docker/data/CASME2_Color_TIM10/'


#def create_flow_input_file(path, pathtxt):
import os	
file1 = open(pathtxt + 'clip1.txt','w') 
file2 = open(pathtxt + 'clip2.txt','w') 
file3 = open(pathtxt + 'clip_flow.txt','w')

folders_sub = os.listdir(path)
for sub in folders_sub:
	print(sub)
	folders_clip = os.listdir(path + sub)
	for clip in folders_clip:
		print(clip)
		pics = os.listdir(path + sub + '/' + clip)

		for pic in pics[:-1]:
			file1.write(path + sub + '/' + clip + '/' + pic + '\n')
			pic = pic.split('.')[0] + '.flo'
			file3.write(path + sub + '/' + clip + '/' + pic + '\n')
		for pic in pics[1:]:
			file2.write(path + sub + '/' + clip + '/' + pic + '\n')

file1.close()
file2.close()
file3.close()
	
#	return True
