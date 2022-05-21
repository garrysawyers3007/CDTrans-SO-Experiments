from math import radians
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import os
import torch.nn.functional as nnf
import torch
import torchvision
import random
import numpy as np
from utils_FDA import FDA_source_to_target_np
from imgaug import augmenters as iaa
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class BaseDataset(object):
    """
    Base class of reid dataset
    """
    ##### revised by luo
    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid, _ in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, style_path=None, adain_transform=None, mode='RGB', patch_size=8):
        self.dataset = dataset
        self.transform = transform
        self.adain_transform = adain_transform
        self.shuffle = ShufflePatches(patch_size)

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        image_net_path =style_path
        f = os.listdir(image_net_path)
        self.style_list = []

        for f1 in f:
            l1 = os.listdir(os.path.join(image_net_path,f1)) 
            self.style_list += [os.path.join(image_net_path,f1,l2) for l2 in l1]
        
        self.art = [index for index, (img_path, pid, camid,trackid, idx) in enumerate(self.dataset) if 'Art' in img_path]
        self.clipart = [index for index, (img_path, pid, camid,trackid, idx) in enumerate(self.dataset) if 'Clipart' in img_path]
        self.product = [index for index, (img_path, pid, camid,trackid, idx) in enumerate(self.dataset) if 'Product' in img_path]
        self.realworld = [index for index, (img_path, pid, camid,trackid, idx) in enumerate(self.dataset) if 'Real_World' in img_path]

    def __len__(self):
        return len(self.dataset)

    def get_aug_full_img(self, index, aug_index):

        path, pid, camid, trackid, idx = self.dataset[index]
        
        style_img = self.loader(random.choice(self.style_list))

        if isinstance(path,tuple):
            img = self.loader(path[0])
        else:
            img = self.loader(path)
        #img.save("./patch_aug_images/images_orig_" + str(index) + ".png", "PNG")

        aug_list = ['FDA','style','adain','weather','cartoon', 'clean'] #['adain','weather','cartoon', 'edged', 'mixup', 'clean', 'style'] 
        aug_type = aug_list[aug_index]

        if aug_type == 'weather':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Snow'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)

        if aug_type == 'cartoon':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Cartoon'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)

        if aug_type == 'style':
            pass

        if aug_type == 'FDA':
            im_src = img
            im_trg = style_img

            im_src = im_src.resize( (224,224), Image.BICUBIC )
            im_trg = im_trg.resize( (224,224), Image.BICUBIC )

            im_src = np.asarray(im_src, np.float32)
            im_trg = np.asarray(im_trg, np.float32)

            im_src = im_src.transpose((2, 0, 1))
            im_trg = im_trg.transpose((2, 0, 1))

            src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

            src_in_trg = src_in_trg.transpose((1,2,0))
            src_in_trg[src_in_trg>255.0] = 255.0
            src_in_trg[src_in_trg<0] = 0
            img = Image.fromarray(src_in_trg.astype(np.uint8))
            #img.save("./patch_aug_images/images_fda_" + str(index) + ".png", "PNG")
            
        if aug_type == 'clean':
            pass

        if aug_type == 'adain':
            pass

        if aug_type == 'edged':
            img_path_split = path.split('/')
            img_path_split[2] = 'office-home-edged'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)
            #img.save(f"./edge_ims_test/{img_path_split[-2]+img_path_split[-1]}")

        if aug_type == 'mixup':
            img_path_split = path.split('/')
            img_path_split[2] = 'office-home-edged'
            img_path = "/".join(img_path_split)
            aug_img = read_image(img_path)
            img = Image.blend(img, aug_img, 0.9)
            #print(path[-1])
            #img.save(f"./figs/{self.alpha}/{img_path_split[-1]}")

        if self.transform is not None:
            if aug_type=='adain' or aug_type=='style':
                img = self.adain_transform(img)
            else:    
                img = self.transform(img)

        return img, aug_type

    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        if 'Art' in img_path:
            rand_indices = random.sample(self.art, 4)
            while index in rand_indices:
                rand_indices = random.sample(self.art, 4)
        if 'Clipart' in img_path:
            rand_indices = random.sample(self.clipart, 4)
            while index in rand_indices:
                rand_indices = random.sample(self.clipart, 4)
        if 'Product' in img_path:
            rand_indices = random.sample(self.product, 4)
            while index in rand_indices:
                rand_indices = random.sample(self.product, 4)
        if 'Real_World' in img_path:
            rand_indices = random.sample(self.realworld, 4)
            while index in rand_indices:
                rand_indices = random.sample(self.realworld, 4)

        rand_img_paths = []
        for rand_index in rand_indices:
            rand_img_path, rand_pid, rand_camid, rand_trackid, rand_idx = self.dataset[rand_index]
            rand_img_paths.append(rand_img_path)

        if isinstance(img_path,tuple):
            all_imgs = []
            all_imgs_path = []
            for i_path in img_path:
                i_img = read_image(i_path)
                if self.transform is not None:
                    i_img = self.transform(i_img)
                    i_img = self.shuffle(i_img)
                all_imgs.append(i_img)
                all_imgs_path.append(i_path)
                # all_imgs_path.append(i_path.split('/')[-1])
            img = tuple(all_imgs)

            # print('data base pid ',pid)
            if isinstance(pid, tuple):
                if isinstance(idx, tuple):
                    return img + pid + (camid, trackid)+ tuple(all_imgs_path)+ idx
                else:
                    return img + pid + (camid, trackid, tuple(all_imgs_path), idx)
            else:
                return img + (pid, camid, trackid, tuple(all_imgs_path), idx)
        else:
            aug_index = random.randint(0, 5)
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            aug_img, aug_type = self.get_aug_full_img(index, aug_index)

            rand_imgs = []
            for i, rand_img_path in enumerate(rand_img_paths):
                rand_img = read_image(rand_img_path)
                if self.transform is not None:
                    rand_img = self.transform(rand_img)
                # rand_aug_img, _ = self.get_aug_full_img(rand_indices[i], aug_index)
                rand_imgs.append(rand_img)
            # if aug_type!='style' or aug_type!='adain':
            #     rand_aug_img = self.shuffle(rand_aug_img)
            #torchvision.utils.save_image(img, f'shuffled/{index}.png')           
            return rand_imgs, aug_img, img, aug_type, pid, camid, trackid, img_path.split('/')[-1],idx

class ShufflePatches(object):
  def __init__(self, patch_size):
    self.ps = patch_size

  def __call__(self, x):
    x = x.unsqueeze(0)
    psize = x.shape[-1]//self.ps
    # divide the batch of images into non-overlapping patches
    u = nnf.unfold(x, kernel_size=psize, stride=psize, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=psize, stride=psize, padding=0)
    return f[0]

class ImageList_style(Dataset):
    def __init__(self, labels=None, transform=None, target_transform=None, mode='RGB',style_path = None):
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        image_net_path =style_path
        f = os.listdir(image_net_path)
        self.style_list = []

        for f1 in f:
            l1 = os.listdir(os.path.join(image_net_path,f1)) 
            self.style_list += [os.path.join(image_net_path,f1,l2) for l2 in l1]

        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        
    def __getitem__(self, index):
        style_img = self.loader(random.choice(self.style_list))

        if self.transform is not None:
            style_img = self.transform(style_img)

        return style_img

    def __len__(self):
        return len(self.style_list)