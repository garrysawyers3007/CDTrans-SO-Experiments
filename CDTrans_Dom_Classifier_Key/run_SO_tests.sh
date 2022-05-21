for target in 'Clipart' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/pretrain.yml' TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2_"$target"_shuffled_all_key_2/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done 

for target in 'Art' 'Product' 'Real_World' #source testing 
do
    
    python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_shuffled_all_key_2/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

done 
