from options.train_options import TrainOptions
import torch

opt = TrainOptions().parse()

a=torch.ones(3,2,2)
print(opt.input_channels)
print(opt.output_channels)

print(opt.input_channels[0])
print(type(opt.input_channels[0]))
b=a[int(opt.input_channels[0]), ...]
print(b)

print(opt.gan_only)
