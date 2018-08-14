

from utils_DTSCNN import augmentation_casme
from list_databases import load_db


###
# data augmentation and sampling as done in the DTSCNN paper
###

#root_db_path = "/home/miha/Documents/ME_data/"
root_db_path = "/media/ostalo/MihaGarafolj/ME_data/"
######################################################

list_dB = ['CASME2_Color_TIM20']
objective_flag = 'st'
spatial_size = 224
r, w, subjects, samples, n_exp, VidPerSubject, vidList, timesteps_TIM, data_dim, channel, table, listOfIgnoredSamples, db_home, db_images, cross_db_flag = load_db(root_db_path, list_dB, spatial_size, objective_flag)
resizedFlag = 0

augmentation_casme(db_images, db_images+"../" + "augmentation/", 500, table, resizedFlag, r, w)
