CUDA_VISIBLE_DEVICES=1 python test.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_cgan --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --gan_only
