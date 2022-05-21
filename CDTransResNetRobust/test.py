import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference, do_inference_uda, do_inference_augs, do_inference_feature_mixup, do_inference_rare_class
from utils.logger import setup_logger
import torch
import torchvision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument("--aug_type", default=None, type=str)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--feature_model_path", default=None, type=str)
    parser.add_argument("--num_patch_wise", default=0, type=int)
    parser.add_argument("--layer_num", default=0, type=int)
    parser.add_argument("--per_class_acc", default=False, type=bool)
    parser.add_argument("--imgnet_model_path", default=None, type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, _, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, args)
    
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    if args.feature_model_path is not None:
        model_feature = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
        model_feature.load_param_finetune(args.feature_model_path)

    model.load_param_finetune(cfg.TEST.WEIGHT)
    if cfg.MODEL.UDA_STAGE == 'UDA':
        do_inference_uda(cfg,
                 model,
                 val_loader,
                 num_query)

    

    elif args.aug_type is not None:
        print("Augs")
        do_inference_augs(cfg,
                    model,
                    val_loader,
                    num_query)
    elif args.feature_model_path is not None:
        print("Feature mixup")
        do_inference_feature_mixup(cfg,
                    model,
                    model_feature,
                    val_loader,
                    num_query,
                    args.layer_num)
    elif args.per_class_acc == True:
        M1 = torchvision.models.resnet50()
        num_ftrs = M1.fc.in_features
        M1.fc = torch.nn.Linear(num_ftrs, num_classes)
        M1.load_state_dict(torch.load(cfg.TEST.WEIGHT))

        M2 = torchvision.models.resnet50()
        num_ftrs = M2.fc.in_features
        M2.fc = torch.nn.Linear(num_ftrs, num_classes)
        M2.load_state_dict(torch.load(args.imgnet_model_path))
        
        print("Per class acc")
        do_inference_rare_class(cfg,
                 M1,
                 M2,
                 val_loader,
                 num_query)
    else:
        print("No Augs")
        do_inference(cfg,
                    model,
                    val_loader,
                    num_query)
