CUDA_VISIBLE_DEVICES=6 python test.py --dataroot /data/mri/data/pdd_sliced --name t123_pdd_L1Only --model pix2pix --which_model_netG unet_128 --which_direction AtoB --dataset_mode aligned_array --norm batch --content_only --loadSize 128 --fineSize 128 --target_type pdd
