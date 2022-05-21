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

for target in 'Art' 'Clipart' 'Real_World'
do
    python train.py --config_file configs/pretrain.yml --resnet_type 'clean' MODEL.DEVICE_ID "('3')"  DATASETS.NAMES 'OfficeHome' \
    OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Product_2'$target'_resnet50_clean' \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Product.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
    MODEL.Transformer_TYPE $model_type \
    MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \
    TEST.IMS_PER_BATCH 64

    python train.py --config_file configs/pretrain.yml --resnet_type 'clean' --only_classifier True MODEL.DEVICE_ID "('3')"  DATASETS.NAMES 'OfficeHome' \
    OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Product_2'$target'_resnet50_clean_only_classifier' \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Product.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
    MODEL.Transformer_TYPE $model_type \
    MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \
    TEST.IMS_PER_BATCH 64

done
