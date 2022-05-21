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
for target in 'Art' 'Clipart' 'Product'
do
python train.py --config_file configs/pretrain.yml --patch_size $patch_size --pretrain_path '../logs/pretrain/deit_base/office-home/Real_World_2'$target'/transformer_best_model.pth' MODEL.DEVICE_ID "('1')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Real_World_2_'$target'SO_Dom_Cls' \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Real_World.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/'$target'.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

done

done