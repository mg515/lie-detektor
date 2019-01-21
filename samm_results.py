

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pandas as pd
import numpy as np
import scipy as sc
from utilities import read_results


print('**** cnn_lstm_id2 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/SAMM_TIM10/Classification/Result/predicts_samm_cnn_lstm_id2.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)



print('**** cnn_lstm_id3 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/SAMM_TIM10/Classification/Result/predicts_samm_cnn_lstm_id3--spatial_size=224.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)
