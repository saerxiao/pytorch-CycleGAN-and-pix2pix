CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_cgan_L1 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --dataset_mode aligned --no_lsgan --norm batch --pool_size 0
