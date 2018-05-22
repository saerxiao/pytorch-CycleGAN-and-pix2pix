## copy from https://github.com/milesial/Pytorch-UNet/blob/master/unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import *

class MyUnetGenerator(nn.Module):
  def __init__(self, input_nc, output_nc, unet_type, gpu_ids=[]):
    super(MyUnetGenerator, self).__init__()
    self.gpu_ids = gpu_ids

    if unet_type == '128':
      model = Unet128(input_nc, output_nc)
    else:
      raise NotImplementedError('Unet type [%s] is not recognized' % unet_type)
    self.model = model
  
  def forward(self, input):
    if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
      return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
    else:
      return self.model(input)
    
  
class Unet128(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Unet128, self).__init__()

        self.inc = inconv(input_nc, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.up1 = up(512+256, 256)
        self.up2 = up(256+128, 128)
        self.up3 = up(128+64, 64)
        self.up4 = up(64+32, 32)
        self.outc = outconv(32, output_nc)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.outc(y)

        return y

