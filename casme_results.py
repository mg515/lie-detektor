

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pandas as pd
import numpy as np
import scipy as sc
from utilities import read_results


print('**** apex1 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_id1.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)



print('**** apex2 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_id2.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** apex2_ponovnoTf ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_id2_ponovnoTf.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)



print('**** apex3 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_id3.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)



# print('**** cnn_lstm_id1 ****')
# path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id1.txt'
# table,acc,f1,cm = read_results(path)

# print(acc)
# print(f1)
# print(cm)


print('**** cnn_lstm_id1_ponovnoTf ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id1_ponovnoTf.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)



# print('**** cnn_lstm_id2 ****')
# path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id2.txt'
# table,acc,f1,cm = read_results(path)

# print(acc)
# print(f1)
# print(cm)


print('**** cnn_lstm_id2_ponovnoTf ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id2_ponovnoTf.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** cnn_lstm_id3 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id3.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** cnn_lstm_id4 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id4.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** cnn_lstm_id5 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id5.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** cnn_lstm_id6 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id6.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** cnn_lstm_id7 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_cnn_lstm_id7.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** c3d_id1 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_c3d_id1.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** VGG_lstm_id1 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Color_TIM10/Classification/Result/predicts_casme2_vgg_lstm_id1.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)

print('**** VGG_lstm_id2 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Optical/Classification/Result/predicts_casme2_vgg_lstm_id2.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('****** CASME 1 & 2 COMBINED ******')
print('**** casme12_cnn_lstm_id2 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME12_Color_TIM10/Classification/Result/predicts_casme12_cnn_lstm_id2.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)

print('**** casme12_cnn_lstm_id7 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME12_Color_TIM10/Classification/Result/predicts_casme12_cnn_lstm_id7.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)
