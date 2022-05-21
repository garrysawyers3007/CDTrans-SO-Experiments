import logging
import numpy as np
import os
import torch.nn.functional as nnf
import time
import torch
import torch.nn as nn
import cv2
from utils.meter import AverageMeter
from utils.metrics import R1_mAP, R1_mAP_eval, R1_mAP_Pseudo, R1_mAP_query_mining, R1_mAP_save_feature, R1_mAP_draw_figure, Class_accuracy_eval
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
import torchvision.transforms as T
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
import random

from datasets.bases import ImageList_style
from torch.utils.data import DataLoader
from pytorch_AdaIN import net
from pytorch_AdaIN.function import adaptive_instance_normalization
from style_augmentation.styleaug import StyleAugmentor

from tensorboardX import SummaryWriter
import tensorflow as tf

import random

writer = SummaryWriter()

def transform_style(resize_size=256, crop_size=224, alexnet=False):
  
  return  T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

def style_transfer(vgg, decoder, content, style, alpha=0.5):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    
    return decoder(feat)

def normalize(X):
    mu = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(X.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(X.device)
    return (X - mu)/std

def shuffle_batch(x, ps):
    psize = x.shape[-1]//ps
    # divide the batch of images into non-overlapping patches
    u = nnf.unfold(x, kernel_size=psize, stride=psize, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=psize, stride=psize, padding=0)
    return f

def do_train_pretrain(cfg,
             model,
             center_criterion,
             train_loader,
             train_loader_clean,
             val_loader,
             val_loader_shuffled,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, patch_size, layer_num, dom_cls=False):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        evaluator = Class_accuracy_eval(dataset=cfg.DATASETS.NAMES, logger=logger)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    best_model_mAP = 0
    min_mean_ent = 1e5

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    vgg.cuda()
    decoder.cuda()
    
    decoder.load_state_dict(torch.load('./pytorch_AdaIN/models/decoder.pth'))
    vgg.load_state_dict(torch.load('./pytorch_AdaIN/models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    augmentor=StyleAugmentor()

    style_data = ImageList_style(transform=transform_style(),style_path = './FDA_dataset_rendition/')
    style_loader = DataLoader(style_data, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True)

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (rand_aug_img, aug_img, img, aug_type, vid, target_cam, target_view, _) in enumerate(train_loader):
            # print('aaaaaa!!!')
            rand_aug_imgs = []
            for i in range(rand_aug_img.shape[1]):
                rand_aug_imgs.append(rand_aug_img[:,i])
            rand_aug_img = torch.stack(rand_aug_imgs, dim=0)
            try:
                inputs_style = iter_source_style.next()
            except:
                iter_source_style = iter(style_loader)
                inputs_style = iter_source_style.next()

            aug_type = np.array(aug_type)
            idx_adain = (aug_type=='adain').nonzero()
            idx_styleaug = (aug_type=='style').nonzero()
            idx = (aug_type!=None).nonzero()

            idx_others = np.array(list((set(idx[0]) - set(idx_adain[0])) - set(idx_styleaug[0])))

            rand_aug_img_adain = rand_aug_img[:, idx_adain[0]].cuda()
            aug_img_adain = aug_img[idx_adain[0]].cuda()
            img_adain = img[idx_adain[0]].cuda()
            vid_adain = vid[idx_adain[0]].cuda()
            target_cam_adain = target_cam[idx_adain[0]].cuda()
            target_view_adain = target_view[idx_adain[0]].cuda()

            rand_aug_img_styleaug = rand_aug_img[:, idx_styleaug[0]].cuda()
            aug_img_styleaug = aug_img[idx_styleaug[0]].cuda()
            img_styleaug = img[idx_styleaug[0]].cuda()
            vid_styleaug = vid[idx_styleaug[0]].cuda()
            target_cam_styleaug = target_cam[idx_styleaug[0]].cuda()
            target_view_styleaug = target_view[idx_styleaug[0]].cuda()

            rand_aug_img_others = rand_aug_img[:, idx_others].cuda()
            aug_img_others = aug_img[idx_others].cuda()
            img_others = img[idx_others].cuda()
            vid_others = vid[idx_others].cuda()
            target_cam_others = target_cam[idx_others].cuda()
            target_view_others = target_view[idx_others].cuda()

            inputs_style_for_adain=inputs_style[:aug_img_adain.size(0)].cuda()
            rand_inputs_style_for_adain=inputs_style[:rand_aug_img_adain.shape[1]].cuda()
            try:
                with torch.no_grad():
                    data_stylized_adain = style_transfer(vgg, decoder, aug_img_adain, inputs_style_for_adain, 0.5)
                    aug_im_restyled_styleaug = augmentor(aug_img_styleaug)
                    
                    rand_data_stylized_adain = []
                    rand_aug_im_restyled_styleaug = []
                    for i in range(rand_aug_img_adain.shape[0]):
                        rand_data_stylized_adain.append(style_transfer(vgg, decoder, rand_aug_img_adain[i], rand_inputs_style_for_adain, 0.5))
                    rand_data_stylized_adain = torch.stack(rand_data_stylized_adain, dim=0)

                    for i in range(rand_aug_img_styleaug.shape[0]):
                        rand_aug_im_restyled_styleaug.append(augmentor(rand_aug_img_styleaug[i]))
                    rand_aug_im_restyled_styleaug = torch.stack(rand_aug_im_restyled_styleaug, dim=0)

                    # rand_data_stylized_adain = shuffle_batch(rand_data_stylized_adain, patch_size)
                    # rand_aug_im_restyled_styleaug = shuffle_batch(rand_aug_im_restyled_styleaug, patch_size)
                
                data_stylized = normalize(torch.cat([data_stylized_adain, aug_im_restyled_styleaug], 0))
                rand_data_stylized = normalize(torch.cat([rand_data_stylized_adain, rand_aug_im_restyled_styleaug], 1))

                rand_aug_img = torch.cat((rand_aug_img_others, rand_data_stylized), dim=1)
                aug_img = torch.cat((aug_img_others, data_stylized), dim=0)
                img = torch.cat((img_others, img_adain, img_styleaug), dim=0)
                vid = torch.cat((vid_others, vid_adain, vid_styleaug), dim=0)
                target_cam = torch.cat((target_cam_others, target_cam_adain, target_cam_styleaug), dim=0)
                target_view = torch.cat((target_view_others, target_view_adain, target_view_styleaug), dim=0)

            except Exception as e:
                rand_aug_img = rand_aug_img_others
                aug_img = aug_img_others
                img = img_others
                vid = vid_others
                target_cam = target_cam_others
                target_view = target_view_others

            W = int(img[0, 0].shape[0]/patch_size)
            H = int(img[0, 0].shape[1]/patch_size)

            final_rand_aug_img = torch.zeros((img.shape[0], 3, 224, 224)).to(device)
            for i in range(patch_size):
                for j in range(patch_size):
                    final_rand_aug_img[:, :, i*W:i*W+W,j*H:j*H+H] = rand_aug_img[random.randint(0, 3), :, :, i*W:i*W+W,j*H:j*H+H]  

            final_rand_aug_img = shuffle_batch(final_rand_aug_img, patch_size)        
            if(len(img)==1):continue
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            aug_img = aug_img.to(device)
            img = img.to(device)
            rand_aug_img = rand_aug_img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                layer_num = random.randint(1,12)
                score, feat = model(final_rand_aug_img, aug_img, target, cam_label=target_cam, view_label=target_view, layer_num=layer_num)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        writer.add_scalar("Train Loss", loss_meter.avg, epoch)
        writer.add_scalar("Train Acc", acc_meter.avg, epoch)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            elif cfg.MODEL.TASK_TYPE == 'classify_DA':
                model.eval()
                for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        output_prob = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
                        evaluator.update((output_prob, vid))
                accuracy,mean_ent = evaluator.compute()
                if (mean_ent < min_mean_ent and not dom_cls) or (best_model_mAP<accuracy and dom_cls):
                    best_model_mAP = accuracy
                    min_mean_ent = mean_ent
                    torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
                logger.info("Classify Domain Adapatation Validation Results - Epoch: {}".format(epoch))
                logger.info("Accuracy: {:.1%} Mean Entropy: {:.1%}".format(accuracy, mean_ent))
                # logger.info("Per-class accuracy: {}".format(acc))
                
                torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

            writer.add_scalar("Test Mean Entropy", mean_ent, epoch)
            writer.add_scalar("Test Acc", accuracy, epoch)       
    # inference
    model.load_param_finetune(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best_model.pth'))
    model.eval()
    evaluator.reset()

    for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                evaluator.update((feat, vid))
            else:
                evaluator.update((feat, vid, camid))
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        accuracy,_ = evaluator.compute()  
        logger.info("Classify Domain Adapatation Validation Results - Best Model")
        logger.info("Accuracy: {:.1%}".format(accuracy))
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Best Model Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    evaluator.reset()

    for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(val_loader_shuffled):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                evaluator.update((feat, vid))
            else:
                evaluator.update((feat, vid, camid))
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        accuracy,_ = evaluator.compute()  
        logger.info("Classify Domain Adapatation Validation Results - Best Model")
        logger.info("Shuffled Accuracy: {:.1%}".format(accuracy))
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Best Model Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    evaluator.reset()

    for n_iter, (rand_img, img, vid, camid, camids, target_view, _) in enumerate(train_loader_clean):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, img, cam_label=camids, view_label=target_view, return_logits=True)
            if cfg.MODEL.TASK_TYPE == 'classify_DA':
                evaluator.update((feat, vid))
            else:
                evaluator.update((feat, vid, camid))
    if cfg.MODEL.TASK_TYPE == 'classify_DA':
        accuracy,_ = evaluator.compute()  
        logger.info("Classify Domain Adapatation Validation Results - Best Model")
        logger.info("Source Clean Accuracy: {:.1%}".format(accuracy))
    else:
        cmc, mAP, _, _, _, _, _ = evaluator.compute()
        logger.info("Best Model Validation Results ")
        logger.info("mAP: {:.1%}".format(mAP))
        for r in [1, 5, 10]:
            logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("reid_baseline.test")
    logger.info("Enter inferencing")
    
    if cfg.TEST.EVAL:
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            evaluator = Class_accuracy_eval(dataset=cfg.DATASETS.NAMES, logger= logger)
        else:
            evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_draw_figure(cfg, num_query, max_rank=50, feat_norm=True,
                       reranking=cfg.TEST.RE_RANKING)
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            
            if cfg.TEST.EVAL:
                if cfg.MODEL.TASK_TYPE == 'classify_DA':
                    probs = model(img, cam_label=camids, view_label=target_view, return_logits=True)
                    evaluator.update((probs, pid))
                else:
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, pid, camid))
            else:
                feat = model(img, cam_label=camids, view_label=target_view)
                evaluator.update((feat, pid, camid, target_view, imgpath))
            img_path_list.extend(imgpath)

    
    if cfg.TEST.EVAL:
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            accuracy, mean_ent = evaluator.compute()  
            logger.info("Classify Domain Adapatation Validation Results - In the source trained model")
            logger.info("Accuracy: {:.1%}".format(accuracy))
            return 
        else:
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results ")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            return cmc[0], cmc[4]
    else:
        print('yes begin saving feature')
        feats, distmats, pids, camids, viewids, img_name_path = evaluator.compute()

        torch.save(feats, os.path.join(cfg.OUTPUT_DIR, 'features.pth'))
        np.save(os.path.join(cfg.OUTPUT_DIR, 'distmat.npy'), distmats)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'label.npy'), pids)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'camera_label.npy'), camids)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'image_name.npy'), img_name_path)
        np.save(os.path.join(cfg.OUTPUT_DIR, 'view_label.npy'), viewids)
        print('over')
