model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='vit_base_patch16_224_TransReID'
    pretrain_model='deit_base_distilled_patch16_224-df68dfff.pth'
else
    model='deit_small'
    model_type='vit_small_patch16_224_TransReID'
    pretrain_model='deit_small_distilled_patch16_224-649709d9.pth'
fi
for target in 'Clipart' 'Product' 'Real_World'  
do
    for num_patch in 5
    do   
    python train.py --config_file configs/pretrain.yml --num_patch_wise $num_patch --layer_num 11 --feature_model_path '../logs/pretrain/deit_base/office-home/Art_2'$target'_augs/transformer_best_model.pth' --lin_prot_path '../logs/pretrain/deit_base/office-home/Art_2_'$target'_only_classifier/transformer_best_model.pth' MODEL.DEVICE_ID "('3')" DATASETS.NAMES 'OfficeHome' \
    OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2'$target'_patch_aug'$num_patch'_'$layer_num'feature_mixup_global_feat' \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
    MODEL.Transformer_TYPE $model_type \
    MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \
    
    done

done 
