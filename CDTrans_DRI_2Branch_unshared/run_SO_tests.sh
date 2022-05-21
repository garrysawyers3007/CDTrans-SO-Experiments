for target in 'Clipart' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Art2"$target"_DRI_2B_freeze_k/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done 

for target in 'Art' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Clipart2"$target"_DRI_2B_freeze_k/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done

for target in 'Art' 'Clipart' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Product2"$target"_DRI_2B_freeze_k/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Product2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Product.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done

for target in 'Art' 'Clipart' 'Product' #source testing 
do
    
    python test.py --config_file 'configs/uda.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/uda/deit_base/office-home/Real_World2"$target"_DRI_2B_freeze_k/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome' DATASETS.NAMES2 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Real_World2'$target'DRI' DATASETS.ROOT_TRAIN_DIR '../data/office-home-aug/Real_World.txt' DATASETS.ROOT_TRAIN_DIR2 '../data/office-home-aug/'$target'.txt' DATASETS.ROOT_TEST_DIR '../data/office-home-aug/'$target'.txt' TEST.IMS_PER_BATCH 32

done