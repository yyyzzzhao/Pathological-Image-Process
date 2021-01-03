''' US dataset for VGG train'''
import os
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def read_txt(files):
    img_paths = []
    labels = []
    with open(files, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            img_paths.append(line[:-2])
            labels.append(int(line[-1]))
    return img_paths, labels

class RESNETdataset(object):
    def __init__(self, dataroot, dataversion, crop_size, phase, augmentation):
        txt_file = os.path.join(dataroot, phase + '.txt')
        self.img_paths, self.labels =  read_txt(txt_file)
        self.crop_size = crop_size
        self.Aug = augmentation
        print(self.img_paths[0])
        print('%s num: %d'%(phase, len(self.img_paths)))

    def __getitem__(self, index):
        img_path = self.img_paths[index]  # G:/DataBase/02_ZS_HCC_pathological/02-data-block/02-img-tiles\#ff0000\A6010 06-02038_Annotation 8_246.png
        idx, _ = os.path.split(img_path)  # A6010 06-02038_Annotation 8_246.png
        idx = idx.split('_')[0]  # s058
        img = Image.open(img_path).convert('RGB')
        # large img input need resize
        img = img.resize((256, 256), Image.BILINEAR)
        clss = self.labels[index]
        # data augmentation
        transform_params = self.get_params(self.crop_size, img.size)
        img_transform = self.get_transform(transform_params, Aug=self.Aug)

        img = img_transform(img)
        return {'data': img, 'label': clss, 'idx': idx}

    def __len__(self):
        return len(self.img_paths)

    def make_dataset(self, dir, max_dataset_size=float("inf")):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def get_params(self, crop_size, load_size):
        w, h = crop_size
        new_h, new_w = load_size


        x = random.randint(0, np.maximum(0, new_w - w))
        y = random.randint(0, np.maximum(0, new_h - h))

        flip = random.random() > 0.5
        return {'crop_pos': (x, y), 'flip': flip}

    def get_transform(self, params, Aug=True):
        transform_list = []
        if params['crop_pos']:
            transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], self.crop_size)))
        if params['flip']:
            transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))
        # add brightness adjust module
        brightness = random.random()
        transform_list += [transforms.ColorJitter(brightness=brightness)]
        transform_list += [transforms.ToTensor()]
        # transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if Aug:
            return transforms.Compose(transform_list)
        else:
            return transforms.Compose([transforms.ToTensor()])

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw, th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
