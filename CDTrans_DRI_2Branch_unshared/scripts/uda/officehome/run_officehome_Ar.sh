model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
for dom_weight in 0.0 
do
for target_dataset in 'Clipart' 'Product' 'Real_World'  # 
do
    python train.py --config_file configs/uda.yml --dom_weight $dom_weight --SO_path '../logs/pretrain/deit_base/office-home/Art_2'$target_dataset'/transformer_best_model.pth' --W_p_path '../logs/pretrain/deit_base/office-home/Art_2_'$target_dataset'SO_Dom_Cls/transformer_best_model.pth' MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR '../logs/uda/'$model'/office-home/Art2'$target_dataset'_DRI_2B_freeze_k_classifier' \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
    DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type \

done

done


