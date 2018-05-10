import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch
#import scipy.misc
#import torchvision.utils as tvutil

class UnalignedArray(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A = np.load(A_path)
        A = torch.from_numpy(A).float()
        B = np.load(B_path)
        if self.opt.output_channels:
          B_select = np.zeros((B.shape[0], B.shape[1], 3))
          for i in range(3):
            B_select[:,:,i] = B[:,:,int(self.opt.output_channels[i])]
          B = B_select
        
        #im_B = scipy.misc.toimage(B)
        #print('tmp/im_B_' + str(index) + '.png')
        #im_B.save('tmp/im_B_' + str(index) + '.png')

        #print("before: {}, {}, {}".format(B[:,:,1].min(), B[:,:,1].max(), B[:,:,1].mean()))
        B = torch.from_numpy(B).float()
        #print("after: {}, {}, {}".format(B[:,:,1].min(), B[:,:,1].max(), B[:,:,1].mean()))
        A = A.permute(2,0,1)
        B = B.permute(2,0,1)
        
        #im_B = scipy.misc.toimage(B.permute(1,2,0).numpy())
        #im_B.save('tmp/im_B_' + str(index) + '.png')
        #tvutil.save_image(im_B, 'tmp/im_B_' + str(index) + '.png')

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        #tvutil.save_image(B, 'tmp/im_B_' + str(index) + '.png')
        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedArray'
