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

for patch_size in 8
do
for target in 'Clipart' 'Product' 'Real_World'
do
for layer_num in 1 
do
python train.py --config_file configs/pretrain.yml --patch_size $patch_size --layer_num $layer_num MODEL.DEVICE_ID "('3')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2_'$target'_DRI_contrastive' \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

done

done

done

