python train_tar.py --batch_size 64 --s 1 --t 0 --dset office-home --output_dir_src '../logs/pretrain/deit_base/office-home/Clipart_2Art_patch_aug5_' --K 3 --KK 2 --gpu_id 0
python train_tar.py --batch_size 64 --s 1 --t 2 --dset office-home --output_dir_src '../logs/pretrain/deit_base/office-home/Clipart_2Product_patch_aug5_' --K 3 --KK 2 --gpu_id 0
python train_tar.py --batch_size 64 --s 1 --t 3 --dset office-home --output_dir_src '../logs/pretrain/deit_base/office-home/Clipart_2Real_World_patch_aug5_' --K 3 --KK 2 --gpu_id 0