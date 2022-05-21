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

for patch_size in 3 4 
do
python train.py --config_file configs/pretrain.yml --patch_size $patch_size --dom_cls True MODEL.DEVICE_ID "('0')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Dom_Cls_shuffled_'$patch_size \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Dom_Cls_train.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Dom_Cls_test.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

done

