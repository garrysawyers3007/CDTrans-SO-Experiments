from PIL import Image, ImageFile

from torch.utils.data import Dataset
import torch
import os.path as osp
import os
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
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        if isinstance(img_path,tuple):
            all_imgs = []
            all_imgs_path = []
            for i_path in img_path:
                i_img = read_image(i_path)
                if self.transform is not None:
                    i_img = self.transform(i_img)
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
            if self.transform is not None:
                img = self.transform(img)
           
            return img, pid, camid, trackid, img_path.split('/')[-1],idx

class AugImageDataset(Dataset):
    def __init__(self, dataset, transform=None, style_path=None, adain_transform=None, aug_type=None, alpha=0.1, mode='RGB', layer_num=-1):
        self.dataset = dataset
        self.transform = transform
        self.adain_transform = adain_transform
        self.aug_type = aug_type
        self.alpha = alpha
        self.layer_num = layer_num

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

    def get_aug_full_img(self, index):

        path, pid, camid, trackid, idx = self.dataset[index]
        #path, target = self.imgs[index]

        #path = path.replace('./data/','../SHOT/object/data/')
        #if 'Product' in path or 'RealWorld' in path:
        #    path = path.replace('data/','../../../SHOT_edge/object/data/')
        style_img = self.loader(random.choice(self.style_list))

        if isinstance(path,tuple):
            img = self.loader(path[0])
        else:
            img = self.loader(path)
        #img.save("./patch_aug_images/images_orig_" + str(index) + ".png", "PNG")

        aug_list = ['FDA','style','adain','weather','cartoon'] #['adain','weather','cartoon', 'edged', 'mixup', 'clean', 'style'] 
        aug_type = self.aug_type if self.aug_type is not None else random.choice(aug_list)

        if aug_type == 'weather':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Snow'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)
            # img = img.resize((224,224))
            # img_arr = np.asarray(img)
            # snow = iaa.imgcorruptlike.Snow(severity=1)
            # frost = iaa.imgcorruptlike.Frost(severity=1)
            # img_arr = snow(image = img_arr)
            # img_arr = frost(image = img_arr)
            # img = Image.fromarray(img_arr.astype(np.uint8))
            #img.save("./patch_aug_images/images_weather_" + str(index) + ".png", "PNG")

        if aug_type == 'cartoon':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Cartoon'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)

            # img_arr = np.asarray(img)
            # cartoon = iaa.Cartoon()
            # img_arr = cartoon(image = img_arr)
            # img = Image.fromarray(img_arr.astype(np.uint8))
            #img.save("./patch_aug_images/images_cartoon_" + str(index) + ".png", "PNG")

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

        if aug_type == 'mixup' and self.alpha<=1.0:
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
                # img_ = img.detach().numpy() * 255
                # img_ = np.transpose(img_, (2,1,0))
                # img_ = Image.fromarray(img_.astype(np.uint8))
                # img_.save("./adain_0.9_imgs/Art/" + path.split('/')[-2]+ path.split('/')[-1], "PNG")
                #torchvision.utils.save_image(img, "./patch_aug_images/images_adain_" + str(index) + ".png")
            else:    
                img = self.transform(img)
                #print(torch.max(img), torch.min(img))
                # img_ = img.numpy() * 255
                # img_ = np.transpose(img_, (2,0,1))
                # img_ = Image.fromarray(img_.astype(np.uint8))
                # img_.save("./patch_aug_images/images_clean_" + str(index) + ".png", "PNG")

        return img, aug_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        if isinstance(img_path,tuple):
            aug_img, aug_type = self.get_aug_full_img(index)
            all_imgs = []
            all_imgs_path = []
            for i_path in img_path:
                i_img = read_image(i_path)
                if self.transform is not None:
                    i_img = self.transform(i_img)
                all_imgs.append(i_img)
                all_imgs_path.append(i_path)
                # all_imgs_path.append(i_path.split('/')[-1])
            img = tuple(all_imgs)

            # print('data base pid ',pid)
            if isinstance(pid, tuple):
                if isinstance(idx, tuple):
                    return (aug_img, aug_type) + img + pid + (camid, trackid)+ tuple(all_imgs_path)+ idx
                else:
                    return (aug_img, aug_type) + img + pid + (camid, trackid, tuple(all_imgs_path), idx)
            else:
                return (aug_img, aug_type) + img + (pid, camid, trackid, tuple(all_imgs_path), idx)
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

            aug_img, aug_type = self.get_aug_full_img(index)
            path_split = img_path.split(".")
            dir_split = path_split[2].split("/")
            dir_split[1] = 'features'
            dir_split[2] = f'combined_{self.layer_num}' if self.layer_num>0 else 'combined_1'
            path_split[2] = "/".join(dir_split)
            path_split[-1] = "npy"
            np_path = ".".join(path_split)

            try:
                with open(np_path, 'rb') as f:
                    feature = np.load(f) 
                    feature = torch.from_numpy(feature)
            except:
                feature = torch.rand(197, 768)
            return aug_img, aug_type, img, feature, pid, camid, trackid, img_path, idx

class AugFeatureImageDataset(Dataset):
    def __init__(self, dataset, transform=None, style_path=None, adain_transform=None, aug_type=None, alpha=0.1, mode='RGB'):
        self.dataset = dataset
        self.transform = transform
        self.adain_transform = adain_transform
        self.aug_type = aug_type
        self.alpha = alpha

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

    def get_aug_full_img(self, index):

        path, pid, camid, trackid, idx = self.dataset[index]
        #path, target = self.imgs[index]

        #path = path.replace('./data/','../SHOT/object/data/')
        #if 'Product' in path or 'RealWorld' in path:
        #    path = path.replace('data/','../../../SHOT_edge/object/data/')
        style_img = self.loader(random.choice(self.style_list))

        if isinstance(path,tuple):
            img = self.loader(path[0])
        else:
            img = self.loader(path)
        #img.save("./patch_aug_images/images_orig_" + str(index) + ".png", "PNG")
        
        aug_list = ['FDA','style','adain','weather','cartoon'] #['adain','weather','cartoon', 'edged', 'mixup', 'clean', 'style'] 
        aug_type = self.aug_type if self.aug_type is not None else random.choice(aug_list)

        if aug_type == 'weather':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Snow'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)
            # img = img.resize((224,224))
            # img_arr = np.asarray(img)
            # snow = iaa.imgcorruptlike.Snow(severity=1)
            # frost = iaa.imgcorruptlike.Frost(severity=1)
            # img_arr = snow(image = img_arr)
            # img_arr = frost(image = img_arr)
            # img = Image.fromarray(img_arr.astype(np.uint8))
            #img.save("./patch_aug_images/images_weather_" + str(index) + ".png", "PNG")

        if aug_type == 'cartoon':
            img_path_split = path.split('/')
            img_path_split[4] = img_path_split[4]+'_Cartoon'
            img_path = "/".join(img_path_split)
            img = read_image(img_path)

            # img_arr = np.asarray(img)
            # cartoon = iaa.Cartoon()
            # img_arr = cartoon(image = img_arr)
            # img = Image.fromarray(img_arr.astype(np.uint8))
            #img.save("./patch_aug_images/images_cartoon_" + str(index) + ".png", "PNG")

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

        if aug_type == 'mixup' and self.alpha<=1.0:
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
                # img_ = img.detach().numpy() * 255
                # img_ = np.transpose(img_, (2,1,0))
                # img_ = Image.fromarray(img_.astype(np.uint8))
                # img_.save("./adain_0.9_imgs/Art/" + path.split('/')[-2]+ path.split('/')[-1], "PNG")
                #torchvision.utils.save_image(img, "./patch_aug_images/images_adain_" + str(index) + ".png")
            else:    
                img = self.transform(img)
                #print(torch.max(img), torch.min(img))
                # img_ = img.numpy() * 255
                # img_ = np.transpose(img_, (2,0,1))
                # img_ = Image.fromarray(img_.astype(np.uint8))
                # img_.save("./patch_aug_images/images_clean_" + str(index) + ".png", "PNG")

        return img, aug_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid,trackid, idx = self.dataset[index]
        if isinstance(img_path,tuple):
            aug_img, aug_type = self.get_aug_full_img(index)
            all_imgs = []
            all_imgs_path = []
            for i_path in img_path:
                i_img = read_image(i_path)
                if self.transform is not None:
                    i_img = self.transform(i_img)
                all_imgs.append(i_img)
                all_imgs_path.append(i_path)
                # all_imgs_path.append(i_path.split('/')[-1])
            img = tuple(all_imgs)

            # print('data base pid ',pid)
            if isinstance(pid, tuple):
                if isinstance(idx, tuple):
                    return (aug_img, aug_type) + img + pid + (camid, trackid)+ tuple(all_imgs_path)+ idx
                else:
                    return (aug_img, aug_type) + img + pid + (camid, trackid, tuple(all_imgs_path), idx)
            else:
                return (aug_img, aug_type) + img + (pid, camid, trackid, tuple(all_imgs_path), idx)
        else:
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)

            aug_imgs = []
            aug_types = []
            aug_list = ['FDA','style','adain','weather','cartoon']
            for aug in aug_list:
                self.aug_type = aug
                aug_img, aug_type = self.get_aug_full_img(index)
                aug_imgs.append(aug_img)
                aug_types.append(aug_type)
            
            return aug_imgs, aug_types, img, pid, camid, trackid, img_path, idx

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
