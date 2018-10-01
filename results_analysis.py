

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pandas as pd
import numpy as np
import scipy as sc

def read_results(path):
	table = pd.read_csv(path, header = None, names = ['subId', 'vidId', 'predict', 'gt'])
	table['vidId'] = table['vidId'].apply(lambda x: x.split('.')[0])
	table['subId'] = table['subId'].apply(lambda x: int(x.split('_')[-1]) + 1)
	print(np.max(table['subId']))

	table_gb = table.groupby(['subId', 'vidId']).agg({'predict': 'first', 'gt': 'first'}).reset_index()

	table_mode = table.groupby(['subId', 'vidId']).apply(lambda x: sc.stats.mode(x.predict)[0][0]).reset_index()
	table_mode.columns.values[2] = 'predict'

	accuracy = accuracy_score(table_gb['gt'], table_mode['predict'])
	f1 = f1_score(table_gb['gt'], table_gb['predict'], average = 'macro')
	cm = confusion_matrix(table_gb['gt'], table_mode['predict'])

	return table,accuracy,f1,cm


print('**** apex1 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_1.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** apex_upsampling ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_upsampling.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)




print('**** apex_flatten_first ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_Cropped/Classification/Result/predicts_casme2_apex_test_flatten.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)


print('**** c3d_1 ****')
path = '/media/ostalo/MihaGarafolj/ME_data/CASME2_TIM10/Classification/Result/predicts_casme2_c3d_1.txt'
table,acc,f1,cm = read_results(path)

print(acc)
print(f1)
print(cm)




