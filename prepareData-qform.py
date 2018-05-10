import glob
import numpy as np
import os
import re
from sets import Set
import nibabel as nib
import scipy.misc
import pickle

scaling = False
destination_dir = "/data/mri/data/qform_sliced-noscaling"
if not os.path.exists(destination_dir):
  os.makedirs(destination_dir)

A_dir = "/data/mri/data/raw-with-dti/resliced_to_dti"
B_dir = "/data/mri/data/raw-with-dti/qform"

def get_subject_id(filepath):
  m = re.match(r'(.*)-\d+-*', os.path.basename(filepath))
  return m.group(1)

def find_range():
  B_files = glob.glob("{}/*-qform.nii.gz".format(B_dir))
  min_025, min_134 = float("inf"), float("inf")
  max_025, max_134 = float("-inf"), float("-inf")
  set1 = [0, 2, 5]
  set2 = [1, 3, 4]
  mins_134, mins_025 = [], []
  maxs_134, maxs_025 = [], []
  mean_134, mean_025 = [], []
  for B_file in B_files:
    print(B_file)
    data = nib.load(B_file).get_data()
    for i in set2:
      current_min = data[:,:,:,i].min()
      mins_134.append(current_min)
      if current_min < min_134:
        min_134 = current_min
      current_max = data[:,:,:,i].max()
      maxs_134.append(current_max)
      if current_max > max_134:
        max_134 = current_max
    for i in set1:
      clipped = np.clip(data[:,:,:,i], 0, np.percentile(data[:,:,:,i], 99.95))
      current_min = clipped.min()
      mins_025.append(current_min)
      if current_min < min_025:
        min_025 = current_min
      current_max = clipped.max()
      maxs_025.append(current_max)
      if current_max > max_025:
        max_025 = current_max
  print("134 min is {}, max is{} \n 025 min is {}, max is {}".format(min_134, max_134, min_025, max_025))
  np.save('qform-134-min', mins_134)
  np.save('qform-134-max', maxs_134)
  np.save('qform-025-min', mins_025)
  np.save('qform-025-max', maxs_025)

def split_data(new_split = True):
  if new_split:
    A_files = glob.glob("{}/*-T1_resliced.nii.gz".format(A_dir))
    subjects = Set()
    for A_file in A_files:
      subjects.add(get_subject_id(A_file))

    B_files = glob.glob("{}/*-qform.nii.gz".format(B_dir))
    N = len(B_files)
    indexes = np.random.permutation(N)
    train_indexes = indexes[0 : int(N*0.7)]
    val_indexes = indexes[int(N*0.7)+1 : int(N*0.8)]
    test_indexes = indexes[int(N*0.8)+1 : N]

    def  save_split(split, files, ids):
      outputfile = "{}/{}_volumes.txt".format(destination_dir, split)
      rlt_files = []
      with open(outputfile, 'w') as f:
        for i in ids:
          if get_subject_id(files[i]) in subjects:
            rlt_files.append(files[i])
            f.write("{}\n".format(files[i]))
      return rlt_files

    train_files = save_split("train", B_files, train_indexes)
    val_files = save_split("val", B_files, val_indexes)
    test_files = save_split("test", B_files, test_indexes)
  else:
    def read_split(split):
      filename = "{}/{}_volumes.txt".format(destination_dir, split)
      with open(filename, 'r') as f:
        content = f.readlines()
      rlt_files = []
      for line in content:
        rlt_files.append(line)
      return rlt_files

    train_files = read_split("train")
    val_files = read_split("val")
    test_files = read_split("test")

  return train_files, val_files, test_files

if scaling:
  with open('qform-minmax.pkl', 'rb') as f:
    rescale_range = pickle.load(f)
  min134 = rescale_range['min134']
  max134 = rescale_range['max134']
  min025 = rescale_range['min025']
  max025 = rescale_range['max025']  

def get_slices(split, B_files, B_key_old=None, B_key_new=None):
  print(split)
  A_to_dir = "{}/{}A".format(destination_dir, split)
  B_to_dir = "{}/{}B".format(destination_dir, split)
  if not os.path.exists(A_to_dir):
    os.makedirs(A_to_dir)
  if not os.path.exists(B_to_dir):
    os.makedirs(B_to_dir)

  def get_A_file(B_file, A_key):
    A_file = os.path.basename(B_file).replace('qform.nii.gz', A_key)
    return "{}/{}".format(A_dir, A_file)
    #return "{}/{}".format(A_dir, re.sub(r'pdd.nii.gz\n', A_key, os.path.basename(B_file)))

  for B_file in B_files:
    if B_key_old and B_key_new:
      B_file = B_file.replace(B_key_old, B_key_new)
    B_file = re.sub(r'nii.gz\n', 'nii.gz', B_file)
    T1_file = get_A_file(B_file, 'T1_resliced.nii.gz')
    T2_file = get_A_file(B_file, 'T2_resliced.nii.gz')
    PD_file = get_A_file(B_file, 'PD_resliced.nii.gz')
    T1_data = nib.load(T1_file).get_data()
    T2_data = nib.load(T2_file).get_data()
    PD_data = nib.load(PD_file).get_data()
    B_data_origin = nib.load(B_file).get_data()

    ## rescale B_data to [0,1]
    B_data = np.zeros(B_data_origin.shape)
    set1, set2 = [0,2,5], [1,3,4]
    if scaling:
      for i in set1:
        B_data[:,:,:,i] = np.clip(B_data_origin[:,:,:,i], 0, np.percentile(B_data_origin[:,:,:,i], 99.95))
        B_data[:,:,:,i] = (B_data[:,:,:,i] - min025) / (max025 - min025)
      for i in set2:
        B_data[:,:,:,i] = (B_data_origin[:,:,:,i] - min134) / (max134 - min134)
    else:
      for i in set1:
        B_data[:,:,:,i] = np.clip(B_data_origin[:,:,:,i], 0, np.percentile(B_data_origin[:,:,:,i], 99.95))
      for i in set2:
        B_data[:,:,:,i] = B_data_origin[:,:,:,i]

    assert T1_data.shape[2]==T2_data.shape[2] and T1_data.shape[2]==PD_data.shape[2] and T1_data.shape[2]==B_data.shape[2]
    subject_id = get_subject_id(B_file)
    print(subject_id)
    for i in range(T1_data.shape[2]):
      t123 = np.zeros((T1_data.shape[0], T1_data.shape[1], 3))
      t123[:,:,0] = T1_data[:,:,i]
      t123[:,:,1] = T2_data[:,:,i]
      t123[:,:,2] = PD_data[:,:,i]
      A_slice = scipy.misc.toimage(t123)
      A_slice_np = (np.array(A_slice.getdata()) / 255.).reshape(A_slice.size[0], A_slice.size[1], 3)
      B_slice_np = B_data[:,:,i,:]
      #AB_np = np.concatenate([A_slice_np, B_slice_np], 1)
      np.save("%s/%s_%04d" % (A_to_dir, subject_id, i), A_slice_np)
      np.save("%s/%s_%04d" % (B_to_dir, subject_id, i), B_slice_np)

#find_range()
train_files, val_files, test_files = split_data(new_split=False)
get_slices('train', train_files, B_key_old='pdd', B_key_new='qform')
get_slices('val', val_files, B_key_old='pdd', B_key_new='qform')
get_slices('test', test_files, B_key_old='pdd', B_key_new='qform')
