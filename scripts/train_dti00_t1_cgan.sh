CUDA_VISIBLE_DEVICES=5 python train.py --dataroot /data/mri/data/raw-with-dti --name dti00_t1_cgan --model pix2pix --which_model_netG unet_128 --which_direction AtoB --dataset_mode dti --no_lsgan --norm batch --pool_size 0 --in_protocal DTI-00 --out_protocal T1 --lambda_A 1 --gan_only
