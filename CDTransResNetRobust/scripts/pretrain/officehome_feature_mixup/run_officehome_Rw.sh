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
python train.py --config_file configs/pretrain.yml --feature_mixup True --aug_type 'adain' MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Real_World_2'$target'_augs' \
DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Real_World.txt' \
DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/Art.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH './data/pretrainModel/'$pretrain_model \
