CUDA_VISIBLE_DEVICES=1 python train.py --dataroot /data/mri/data/pdd_sliced --name t123_pdd_cgan_L1 --model pix2pix --which_model_netG unet_128 --which_direction AtoB --dataset_mode aligned_array --no_lsgan --norm batch --pool_size 0 --lambda_A 1 --loadSize 128 --fineSize 128
