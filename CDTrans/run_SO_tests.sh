# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' --per_class_acc True --imgnet_model_path '../logs/pretrain/deit_base/office-home/Art_2_'$target'_only_classifier/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../remote_logs/pretrain/deit_base/office-home/Art_2"$target"_patch_aug5/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' --per_class_acc True --imgnet_model_path '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_only_classifier/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../remote_logs/pretrain/deit_base/office-home/Clipart_2"$target"_patch_aug5_/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

# done 

# for target in 'Art' 'Clipart' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' --per_class_acc True --imgnet_model_path '../logs/pretrain/deit_base/office-home/Product_2_'$target'_only_classifier/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Product_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Product_2_'$target'_only_classifier' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Product.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

# done 

# for target in 'Art' 'Clipart' 'Product' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' --per_class_acc True --imgnet_model_path '../logs/pretrain/deit_base/office-home/Real_World_2_'$target'_only_classifier/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Real_World_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Real_World_2_'$target'_only_classifier' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Real_World.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32

# done 


# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2_"$target"_edged_100%_key/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Art.txt' TEST.IMS_PER_BATCH 32
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2_"$target"_edged_100%_query/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Art.txt' TEST.IMS_PER_BATCH 32
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2_"$target"_edged_100%_value/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Art.txt' TEST.IMS_PER_BATCH 32

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_edged_100%_key/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Clipart.txt' TEST.IMS_PER_BATCH 32
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_edged_100%_query/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Clipart.txt' TEST.IMS_PER_BATCH 32
#     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_edged_100%_value/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Clipart.txt' TEST.IMS_PER_BATCH 32

# done 


# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
    
#     python test.py --config_file 'configs/pretrain.yml' --feature_model_path '../logs/pretrain/deit_base/office-home/Clipart_2Art_augs/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.1_6_clean/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt' TEST.IMS_PER_BATCH 32
#     python test.py --config_file 'configs/pretrain.yml' --feature_model_path '../logs/pretrain/deit_base/office-home/Clipart_2Art_augs/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.1_6_clean/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' TEST.IMS_PER_BATCH 32
        # python test.py --config_file 'configs/pretrain.yml' --feature_model_path '../logs/pretrain/deit_base/office-home/Clipart_2Art_augs/transformer_best_model.pth' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.1_"$N"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_temp' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt'
    # python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.01/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_feature_mixup_0.01' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt' 
    # python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.01_11/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_feature_mixup_0.01_11' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
    # python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.1/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_feature_mixup_0.1' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
    #     python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_feature_mixup_0.1_"$N"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_feature_mixup_0.1_'$N DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
    # python test.py --config_file 'configs/pretrain.yml' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2_"$target"_freeze_MHSA_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2_'$target'_freeze_MHSA_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'

# done

#Cartoon
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'cartoon' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#Snow
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'weather' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#AdaIN
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'adain' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#FDA
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'FDA' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done


#Edged
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'edged' MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#Mixup(alpha=0.1)
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.1 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#Mixup(alpha=0.5)
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.5 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

#Mixup(alpha=0.9)
# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"_augs/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target'_augs' DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done

# for target in 'Clipart' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Art_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Art_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done 

# for target in 'Art' 'Product' 'Real_World' #source testing 
# do
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Clipart.txt'
#     python test.py --config_file 'configs/pretrain.yml' --aug_type 'mixup' --alpha 0.9 MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('../logs/pretrain/deit_base/office-home/Clipart_2"$target"/transformer_best_model.pth')" DATASETS.NAMES 'OfficeHome'  OUTPUT_DIR '../logs/pretrain/deit_base/office-home/Clipart_2'$target DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt' 

# done
 