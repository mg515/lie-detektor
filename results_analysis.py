


from utilities import read_results



# path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical_Aug/Classification/Result/predicts_predicts_casme2_optical_withDroupout05.txt'
# table,acc,cm = read_results(path)

#path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical_TIM20_Aug/Classification/Result/predicts_casme2_tim20_optical_aug.txt'
#table,acc,cm = read_results(path)


#path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical2_Aug/Classification/Result/predicts_casme2_optical2_aug.txt'
#table,acc,cm = read_results(path)


#path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical_Aug/Classification/Result/predicts_casme2_optical_aug.txt'
#table,acc,cm = read_results(path)

path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_1.txt'
table,acc,f1,cm = read_results(path)

print(table)
print(acc)
print(f1)
print(cm)




