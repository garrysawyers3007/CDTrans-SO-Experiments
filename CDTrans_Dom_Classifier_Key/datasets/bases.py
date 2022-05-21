from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import torch.nn.functional as nnf
import torch
import torchvision
import random

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
    def __init__(self, dataset, transform=None, patch_size=1):
        self.dataset = dataset
        self.transform = transform
        self.shuffle = ShufflePatches(patch_size)
        self.indices = [i for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        rand_indices = random.sample(self.indices, 4)
        while index in rand_indices:
            rand_indices = random.sample(self.indices, 4)

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
            img = read_image(img_path)
            rand_img = read_image(rand_img_path)
            if self.transform is not None:
                img = self.transform(img)
            rand_imgs = []
            for i, rand_img_path in enumerate(rand_img_paths):
                rand_img = read_image(rand_img_path)
                if self.transform is not None:
                    rand_img = self.transform(rand_img)
                rand_imgs.append(rand_img)
            #torchvision.utils.save_image(img, f'shuffled/{index}.png')           
            return rand_imgs, img, pid, camid, trackid, img_path.split('/')[-1],idx

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