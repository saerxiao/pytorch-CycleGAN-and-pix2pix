import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import numpy as np

class AlignedArray(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = np.load(AB_path) ## AB is a numpy array in the range of [0,1]
        AB = torch.from_numpy(AB).float() ## shape: fineSize x (2 * fineSize) x 3
        
        AB = AB.permute(2,0,1)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        A_original = A.clone()
        if self.opt.input_channels:
          if len(self.opt.input_channels) == 1:
            A[0, ...] = A_original[int(self.opt.input_channels[0]), ...]
            A[1, ...] = A[0, ...] 
            A[2, ...] = A[0, ...]
          elif len(self.opt.input_channels) == 2:
            A[0, ...] = A_original[int(self.opt.input_channels[0]), ...]
            A[1, ...] = A_original[int(self.opt.input_channels[1]), ...]
            A[2, ...] = A_original[0, ...] * 0.5 + A_original[1, ...] * 0.5
        
        if self.opt.output_channels:
          # assume output only has at most one channel
          if len(self.opt.output_channels) == 1:
            B[0, ...] = A_original[int(self.opt.output_channels[0]), ...]
            B[1, ...] = B[0, ...]
            B[2, ...] = B[0, ...] 

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedArray'
