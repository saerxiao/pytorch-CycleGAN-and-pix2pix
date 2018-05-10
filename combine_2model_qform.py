import glob
import numpy as np
import os

qform025_dir = 'results/t123_qform025_perceptOnly/test_85/numpy'
files = glob.glob("{}/*.npy".format(qform025_dir))

destination_dir = 'results/t123_qform_025percept_134cganL1/numpy'
if not os.path.exists(destination_dir):
  os.makedirs(destination_dir)


for file025 in files:
  file134 = file025.replace('qform025_perceptOnly/test_85', 'qform134_cgan_L1/test_latest')
  np025 = np.load(file025)
  np134 = np.load(file134)
  output = np.zeros((128,128,6))
  output[:,:,0] = np025[:,:,0]
  output[:,:,2] = np025[:,:,1]
  output[:,:,5] = np025[:,:,2]
  output[:,:,1] = np134[:,:,0]
  output[:,:,3] = np134[:,:,1]
  output[:,:,4] = np134[:,:,2]
  filename = os.path.basename(file025)
  output_path = "{}/{}".format(destination_dir, filename)
  print(output_path)
  np.save(output_path, output)


