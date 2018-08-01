


from utilities import read_results


# 
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical_Aug/Classification/Result/predicts_casme2_optical_aug_c3dwithPadding.txt'
table,acc,cm = read_results(path)


path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical_Aug/Classification/Result/predicts_casme2_optical_augmentationEachFrame_c3d.txt'
table,acc,cm = read_results(path)








