CUDA_VISIBLE_DEVICES=4 python train.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t2_t1_cgan_L1 --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --no_lsgan --norm batch --pool_size 0 --input_channels 1 --output_channels 0 --lambda_A 100
