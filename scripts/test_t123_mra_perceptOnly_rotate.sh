CUDA_VISIBLE_DEVICES=3 python test.py --dataroot /data/mri/data/multi-pix2pix-pytorch/t123_mra --name t123_mra_perceptOnly --model pix2pix --which_model_netG unet_256 --which_direction AtoB --dataset_mode aligned --norm batch --content_only --content_loss_type percept --random_rotation --how_many 100
