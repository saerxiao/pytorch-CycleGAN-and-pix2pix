CUDA_VISIBLE_DEVICES=2 python test.py --dataroot /data/mri/data/color_fa_sliced/t123_colorfa --name t123_colorfa_perceptOnly_mul255 --model pix2pix --which_model_netG unet_128 --which_direction AtoB --dataset_mode aligned --norm batch --content_only --content_loss_type percept --loadSize 128 --fineSize 128
