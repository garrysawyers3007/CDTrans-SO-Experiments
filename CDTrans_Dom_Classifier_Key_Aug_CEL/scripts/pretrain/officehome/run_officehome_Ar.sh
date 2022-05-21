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

for layer_num in 1 2 3 4 5 6
do
python train.py --config_file configs/pretrain.yml --layer_num $layer_num MODEL.DEVICE_ID "('2')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2_Clipart_shuffled_CEL_layer_'$layer_num \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Clipart.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

python train.py --config_file configs/pretrain.yml --layer_num $layer_num MODEL.DEVICE_ID "('2')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2_Product_shuffled_CEL_layer_'$layer_num \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Product.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

python train.py --config_file configs/pretrain.yml --layer_num $layer_num MODEL.DEVICE_ID "('2')" DATASETS.NAMES 'OfficeHome' \
OUTPUT_DIR '../logs/pretrain/'$model'/office-home/Art_2_Real_World_shuffled_CEL_layer_'$layer_num \
DATASETS.ROOT_TRAIN_DIR '../data/OfficeHomeDataset/Art.txt' \
DATASETS.ROOT_TEST_DIR '../data/OfficeHomeDataset/Real_World.txt'   \
MODEL.Transformer_TYPE $model_type \
MODEL.PRETRAIN_PATH '../data/pretrainModel/'$pretrain_model \

done

