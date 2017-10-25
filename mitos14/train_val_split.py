import glob 
import random 
import os
import subprocess

random.seed(12)
input_path = '/home/wangsq/data_set/mitos14/new_split/training/*jpg'
val_path = '/home/wangsq/data_set/mitos14/new_split/validation/'

input_array = glob.glob(input_path)
train_size = len(input_array)
input_index = range(train_size)
random.shuffle(input_index)
val_index = random.sample(input_index, 200)

for i in val_index:
	name = input_array[i]
	filepath, filename = os.path.split(name)
	filtername, exts = os.path.splitext(filename)

  	mitosis = filepath + '/' + filtername + '_mitosis.csv'
  	
  	non_mitosis = filepath + '/' + filtername + '_not_mitosis.csv' 
	
	subprocess.call('mv ' + name + ' ' + val_path, shell = True)  
	subprocess.call('mv ' + mitosis + ' ' + val_path, shell = True)  
	subprocess.call('mv ' + non_mitosis + ' ' + val_path, shell = True)  