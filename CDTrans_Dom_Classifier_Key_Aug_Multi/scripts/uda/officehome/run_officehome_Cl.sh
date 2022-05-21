model=$1
if [ ! -n "$1" ]
then 
    echo 'pelease input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('3')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
for dom_weight in 0.0 0.2 1.0
do
for target_dataset in 'Art' 'Product' 'Real_World'
do
    python train.py --config_file configs/uda.yml --dom_weight $dom_weight MODEL.DEVICE_ID $gpus  \
    OUTPUT_DIR '../logs/uda/'$model'/office-home/Clipart2'$target_dataset'DRI_'$dom_weight \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Clipart.txt' \
    DATASETS.ROOT_TRAIN_DIR2 '../data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type \

done

done

