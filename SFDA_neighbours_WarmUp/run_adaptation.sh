python train_tar.py --batch_size 64 --s 0 --t 1 --dset OfficeHomeDataset --output_dir_src '../remote_logs/pretrain/deit_base/office-home/Art_2Clipart_patch_aug5' --K 6 --KK 4 --gpu_id 3
python train_tar.py --batch_size 64 --s 0 --t 2 --dset OfficeHomeDataset --output_dir_src '../remote_logs/pretrain/deit_base/office-home/Art_2Product_patch_aug5' --K 6 --KK 4 --gpu_id 3
python train_tar.py --batch_size 64 --s 0 --t 3 --dset OfficeHomeDataset --output_dir_src '../remote_logs/pretrain/deit_base/office-home/Art_2Real_World_patch_aug5' --K 6 --KK 4 --gpu_id 3