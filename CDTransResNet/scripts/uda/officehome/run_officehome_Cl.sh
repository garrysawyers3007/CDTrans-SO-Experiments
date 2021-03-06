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
for target_dataset in 'Real_World'
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus  \
    OUTPUT_DIR '../logs/uda/'$model'/office-home/Clipart_3'$target_dataset \
    MODEL.PRETRAIN_PATH '../logs/pretrain/'$model'/office-home/Clipart_3/transformer_best_model.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Clipart.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/OfficeHomeDataset/Clipart.txt' \
    DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type \

done

