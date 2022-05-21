from email.policy import default
import os
from xml.etree.ElementTree import TreeBuilder
from torch.backends import cudnn
from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, create_scheduler
from loss import make_loss
from processor import do_train_pretrain, do_train_pretrain_augs, do_train_uda, do_train_pretrain_preload, do_create_mixup_dataset, do_train_pretrain_patch_augs
import random
import torch
import numpy as np
import os
import argparse
import torchvision
# from timm.scheduler import create_scheduler

from config import cfg
from timm.data import Mixup
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def corrected_state_dict(state_dict):
    return {k.replace('module.model.',''):v for k,v in state_dict.items() if "attacker" not in k and "new_mean" not in k and "new_std" not in k}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
    #                     help='LR scheduler (default: "cosine"')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
    #                     help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
    #                     help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
    #                     help='learning rate noise std-dev (default: 1.0)')
    # parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    # parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
    #                     help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
    #                     help='LR decay rate (default: 0.1)')
    # parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--augs", default=False, type=bool)
    parser.add_argument("--freeze_4", default=False, type=bool)
    parser.add_argument("--freeze_MHSA", default=False, type=bool)
    parser.add_argument("--preload_path", default=None, type=str)
    parser.add_argument("--feature_mixup", default=False, type=bool)
    parser.add_argument("--aug_type", default=None, type=str)
    parser.add_argument("--feature_model_path", default=None, type=str)
    parser.add_argument("--num_patch_wise", default=0, type=int)
    parser.add_argument("--layer_num", default=0, type=int)
    parser.add_argument("--only_classifier", default=False, type=bool)
    parser.add_argument("--resnet_type", default=None, type=str)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    else:
        pass

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.MODEL.UDA_STAGE == 'UDA':
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, train_loader1, train_loader2, img_num1, img_num2, s_dataset, t_dataset = make_dataloader(cfg)
    else:
        train_loader, train_loader_normal, val_loader, train_val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, args)
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    if args.preload_path is not None:
        model.load_state_dict(torch.load(args.preload_path))
    layers_to_freeze = list(model.state_dict().keys())[4:52]

    if args.freeze_4==True:
        for k, v in model.named_parameters():
            #print(k, k in layers_to_freeze)
            if k in layers_to_freeze:
                v.requires_grad = False
    
    if args.freeze_MHSA == True:
        for k,v in model.base.named_parameters():
            if 'attn' not in k:
                v.requires_grad=False

    # if args.only_classifier == True:
    #     for k,v in model.base.named_parameters():
    #         v.requires_grad=False

    # if args.freeze_MHSA==True:
    #     for k, v in model.named_parameters():
    #         print(k, v.requires_grad)

    if args.resnet_type == 'robust':
        model = torchvision.models.resnet50()
        model.load_state_dict(corrected_state_dict(torch.load('../data/pretrainModel/resnet-50-l2-eps0.1.ckpt')['model']))
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)

        if args.only_classifier == True:
            for k, v in model.named_parameters():
                if 'fc' not in k:
                    v.requires_grad = False

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    if args.num_patch_wise>0:
        model_feature = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
        model_feature.load_param_finetune(args.feature_model_path)
    
    if cfg.MODEL.UDA_STAGE == 'UDA':
        do_train_uda(
        cfg,
        model,
        center_criterion,
        train_loader,
        train_loader1,
        train_loader2,
        img_num1,
        img_num2,
        val_loader,
        s_dataset, s_dataset,
        optimizer,
        optimizer_center,
        scheduler,  
        loss_func,
        num_query, args.local_rank
    )
    else:
        if args.feature_mixup == True:
            print('feature mixup')
            do_create_mixup_dataset(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,  
                loss_func,
                num_query, args.local_rank, args.layer_num
            )
        elif args.augs==True:
            print('pretrain train augs')
            do_train_pretrain_augs(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                train_val_loader,
                optimizer,
                optimizer_center,
                scheduler,  
                loss_func,
                num_query, args.local_rank
            )
        elif args.preload_path is not None:
            print('pretrain train preload')
            do_train_pretrain_preload(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,  
                loss_func,
                num_query, args.local_rank
            )
        elif args.num_patch_wise>0:
            print("patch wise training")
            do_train_pretrain_patch_augs(
                cfg,
                model,
                model_feature,
                center_criterion,
                train_loader,
                val_loader,
                train_val_loader,
                optimizer,
                optimizer_center,
                scheduler,
                loss_func,
                num_query, args.local_rank,args.num_patch_wise, args.layer_num)
        else:
            print('pretrain train')
            do_train_pretrain(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,  
                loss_func,
                num_query, args.local_rank, args.layer_num
            )

    
    
