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
    # python train.py --config_file configs/pretrain.yml MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'OfficeHome' \
    # OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2'$target \
    # DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Art.txt' \
    # DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target'.txt'   \
    # MODEL.Transformer_TYPE $model_type \
    # MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \

    python train.py --config_file configs/pretrain.yml --augs True --aug_type 'edged' MODEL.DEVICE_ID "('3')" DATASETS.NAMES 'OfficeHome' \
    OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2_'$target'_edged_100%_query' \
    DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
    DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
    MODEL.Transformer_TYPE $model_type \
    MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \
    TEST.IMS_PER_BATCH 64

done 


