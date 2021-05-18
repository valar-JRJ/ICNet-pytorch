import os
import collections
import random
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as tf

from torch.utils import data

from utils.utils import recursive_glob
# from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class SUNRGBDLoader(data.Dataset):
    """SUNRGBD loader

    Download From:
    http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
        test source: http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-test_images.tgz
        train source: http://www.doc.ic.ac.uk/~ahanda/SUNRGBD-train_images.tgz

        first 5050 in this is test, later 5051 is train
        test and train labels source:
        https://github.com/ankurhanda/sunrgbd-meta-data/raw/master/sunrgbd_train_test_labels.tar.gz
    """

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=(480, 640),
        img_norm=False,
        test_mode=False,
    ):
        self.root = root
        self.is_transform = is_transform
        self.n_classes = 38
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.anno_files = collections.defaultdict(list)
        self.cmap = self.color_map(normalized=False)

        split_map = {"training": "train", "val": "test"}
        self.split = split_map[split]

        if not self.test_mode:
            self.images_base = os.path.join(self.root, self.split)
            self.annotations_base = os.path.join(self.root, "labels", self.split)
            print(self.images_base)
            print(self.annotations_base)

            # for split in ["train", "test"]:
            file_list = sorted(recursive_glob(rootdir=self.images_base, suffix="jpg"))
            self.files[self.split] = file_list
            # print(self.files[self.split])

            # for split in ["train", "test"]:
            file_list = sorted(recursive_glob(rootdir=self.annotations_base, suffix="png"))
            self.anno_files[self.split] = file_list
            # print(self.anno_files[self.split])

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = self.anno_files[self.split][index].rstrip()
        # print(img_path)
        # print(lbl_path)
        # img_number = img_path.split('/')[-1]
        # lbl_path = os.path.join(self.root, 'annotations', img_number).replace('jpg', 'png')

        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        # print(np.unique(lbl))

        if not (len(img.shape) == 3 and len(lbl.shape) == 2):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        # augmentations
        img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        # img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = np.array(Image.fromarray(img).resize((self.img_size[0], self.img_size[1])))
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        # print(np.unique(lbl.astype(int)))
        lbl = np.array(Image.fromarray(lbl).resize((self.img_size[0], self.img_size[1]), Image.NEAREST))
        lbl = lbl.astype(int)
        # assert np.all(classes == np.unique(lbl))
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def color_map(self, N=256, normalized=False):
        """
        Return Color Map in PASCAL VOC format
        """

        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        dtype = "float32" if normalized else "uint8"
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7 - j)
                g = g | (bitget(c, 1) << 7 - j)
                b = b | (bitget(c, 2) << 7 - j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = cmap / 255.0 if normalized else cmap
        return cmap

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.cmap[l, 0]
            g[temp == l] = self.cmap[l, 1]
            b[temp == l] = self.cmap[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def augmentations(self, img, mask):
        gamma = 0.5
        hflip = 0.5
        vflip = 0.5
        degree = 60
        crop_size = 480
        base_size = 520
        PIL2Numpy = False

        if isinstance(img, np.ndarray):
          img = Image.fromarray(img, mode="RGB")
          mask = Image.fromarray(mask, mode="L")
          PIL2Numpy = True
          # print('1 {}'.format(np.unique(mask)))

        # gamma
        # assert img.size == mask.size
        # img = tf.adjust_gamma(img, random.uniform(1, 1 + gamma))

        # random scale (short edge)
        short_size = random.randint(int(base_size * 0.7), int(base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        #
        # # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # horizontal flip
        if random.random() < hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            # print('hflip {}'.format(np.unique(mask)))

        # vertical flip
        if random.random() < vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            # print('vflip {}'.format(np.unique(mask)))

        # random rotate
        # rotate_degree = random.random() * 2 * degree - degree
        # img = tf.affine(
        #     img,
        #     translate=(0, 0),
        #     scale=1.0,
        #     angle=rotate_degree,
        #     resample=Image.BILINEAR,
        #     fillcolor=(0, 0, 0),
        #     shear=0.0,
        # )
        # mask = tf.affine(
        #     mask,
        #     translate=(0, 0),
        #     scale=1.0,
        #     angle=rotate_degree,
        #     resample=Image.NEAREST,
        #     # fillcolor=250,
        #     shear=0.0,
        # )
        # print('rotate {}'.format(np.unique(mask)))
        
        if PIL2Numpy:
          img, mask = np.array(img), np.array(mask, dtype=np.uint8)
          # print('2 {}'.format(np.unique(mask)))

        return img, mask


if __name__ == "__main__":
    # pass
    import matplotlib.pyplot as plt

    # augmentations = Compose([Scale(512), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "../../data/samples"
    dst = SUNRGBDLoader(local_path, is_transform=True, img_size=(530, 530), img_norm=True)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
