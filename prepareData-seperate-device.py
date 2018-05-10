import glob
import numpy as np
import os
import re
from sets import Set
import nibabel as nib
import scipy.misc

destination_dir = "/data/mri/data/res256-seperate-device"  #"/data/mri/data/color_fa_sliced-seperate-device"
if not os.path.exists(destination_dir):
  os.makedirs(destination_dir)

A_dir = "/data/mri/data/res256/all"  #"/data/mri/data/raw-with-dti/resliced_to_dti"
B_dir = A_dir  #"/data/mri/data/raw-with-dti/color_fa"

key = "MRA_resliced"  #'color_fa'
B_to_dir_key = 'mra' #color_fa

#HH_files = glob.glob("{}/*-HH-*-T1_resliced.nii.gz".format(A_dir))
#Guys_files = glob.glob("{}/*-Guys-*-T1_resliced.nii.gz".format(A_dir))
#print(len(HH_files))  ## used for test
#print(len(Guys_files)) ## used for training and validate


def get_subject_id(filepath):
  m = re.match(r'(.*)-\d+-*', os.path.basename(filepath))
  return m.group(1)

def split_data(new_split = True):
  if new_split:
    A_files = glob.glob("{}/*-T1_resliced.nii.gz".format(A_dir))
    subjects = Set()
    for A_file in A_files:
      subjects.add(get_subject_id(A_file))

    B_guys_files = glob.glob("{}/*-Guys-*-{}.nii.gz".format(B_dir, key))
    N = len(B_guys_files)
    indexes = np.random.permutation(N)
    train_indexes = indexes[0 : int(N*0.9)]
    val_indexes = indexes[int(N*0.9)+1 : N]

    B_HH_files = glob.glob("{}/*-HH-*-{}.nii.gz".format(B_dir, key))
    test_indexes = range(len(B_HH_files))

    def  save_split(split, files, ids):
      outputfile = "{}/{}_volumes.txt".format(destination_dir, split)
      rlt_files = []
      with open(outputfile, 'w') as f:
        for i in ids:
          if get_subject_id(files[i]) in subjects:
            rlt_files.append(files[i])
            f.write("{}\n".format(files[i]))
      return rlt_files

    train_files = save_split("train", B_guys_files, train_indexes)
    val_files = save_split("val", B_guys_files, val_indexes)
    test_files = save_split("test", B_HH_files, test_indexes)
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

def get_slices(split, B_files):
  print(split)
  A_to_dir = "{}/t123/{}".format(destination_dir, split)
  B_to_dir = "{}/{}/{}".format(destination_dir, B_to_dir_key, split)
  if not os.path.exists(A_to_dir):
    os.makedirs(A_to_dir)
  if not os.path.exists(B_to_dir):
    os.makedirs(B_to_dir)

  def get_A_file(B_file, A_key):
    A_file = "{}/{}".format(A_dir, os.path.basename(B_file).replace("{}.nii.gz".format(key), A_key))
    #A_file = "{}/{}".format(A_dir, re.sub(r'color_fa.nii.gz', A_key, os.path.basename(B_file)))
    A_file = re.sub(r'nii.gz\n', 'nii.gz', A_file)
    return A_file

  cnt = 0
  for B_file in B_files:
    T1_file = get_A_file(B_file, 'T1_resliced.nii.gz')
    T2_file = get_A_file(B_file, 'T2.nii.gz')
    PD_file = get_A_file(B_file, 'PD_resliced.nii.gz')
    T1_data = nib.load(T1_file).get_data()
    T2_data = nib.load(T2_file).get_data()
    PD_data = nib.load(PD_file).get_data()
    B_file = re.sub(r'nii.gz\n', 'nii.gz', B_file)
    B_data = nib.load(B_file).get_data()

    subject_id = get_subject_id(B_file)
    if T1_data.shape[2]==T2_data.shape[2] and T1_data.shape[2]==PD_data.shape[2] and T1_data.shape[2]==B_data.shape[2]:
      print(subject_id)
      cnt = cnt + 1
      for i in range(T1_data.shape[2]):
        t123 = np.zeros((T1_data.shape[0], T1_data.shape[1], 3))
        t123[:,:,0] = T1_data[:,:,i]
        t123[:,:,1] = T2_data[:,:,i]
        t123[:,:,2] = PD_data[:,:,i]
        A_slice = scipy.misc.toimage(t123)
        B_slice = scipy.misc.toimage(B_data[:,:,i])
        #B_slice = scipy.misc.toimage(B_data[:,:,i,:])
        A_slice.save("%s/%s_%04d.png" % (A_to_dir, subject_id, i))
        B_slice.save("%s/%s_%04d.png" % (B_to_dir, subject_id, i))
    else:
      print("{}: shapes are different, skip".format(subject_id))
  print("{} volumes for {}".format(cnt, split))

train_files, val_files, test_files = split_data(new_split=False)
get_slices('train', train_files)
#get_slices('val', val_files)
get_slices('test', test_files)

