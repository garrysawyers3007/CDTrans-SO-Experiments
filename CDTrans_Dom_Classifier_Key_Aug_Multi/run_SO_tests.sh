for target in 'Clipart' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Art2"$target"DRI/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done 

for target in 'Art' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Clipart2"$target"DRI/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done 

 