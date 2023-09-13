
import csv
import torchvision.transforms
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from typing import Callable, Iterable, Tuple
from pathlib import Path
import torch.nn.functional as F


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


CIFAR_PATH = Path("./data_cifar10")
CIFAR_TRANSFORM_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_TRANSFORM_NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
CIFAR_TRANSFORM_NORMALIZE = transforms.Normalize(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_NORMALIZE_INV = NormalizeInverse(
    CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD
)
CIFAR_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
#CIFAR_TRANSFORM_TRAIN_XY = lambda xy: (CIFAR_TRANSFORM_TRAIN(xy[0]), xy[1])

CIFAR_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.ToTensor(),
        CIFAR_TRANSFORM_NORMALIZE,
    ]
)
CIFAR_TENSOR_TRAIN = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

GTSRB_TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]
)

GTSRB_TRANSFORM_TRAIN = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


imagenet_transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])

imagenet_transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])

To_Tensor = transforms.ToTensor()

#CIFAR_TRANSFORM_TEST_XY = lambda xy: (CIFAR_TRANSFORM_TEST(xy[0]), xy[1])

CIFAR_TRANSFORM_TEST_XY = lambda xyz: (CIFAR_TRANSFORM_TEST(xyz[0]), xyz[1], xyz[2])
CIFAR_TRANSFORM_TRAIN_XY = lambda xyz: (CIFAR_TRANSFORM_TRAIN(xyz[0]), xyz[1], xyz[2])

GTSRB_TRANSFORM_TEST_XY = lambda xyz: (GTSRB_TRANSFORM_TEST(xyz[0]), xyz[1], xyz[2])
GTSRB_TRANSFORM_TRAIN_XY = lambda xyz: (GTSRB_TRANSFORM_TRAIN(xyz[0]), xyz[1], xyz[2])


IMAGENET_TRANSFORM_TEST_XY = lambda xyz: (imagenet_transform_test(xyz[0]), xyz[1], xyz[2])
IMAGENET_TRANSFORM_TRAIN_XY = lambda xyz: (imagenet_transform_train(xyz[0]), xyz[1], xyz[2])


CIFAR_TRANSFORM_TEST_XY_TWO = lambda xyz: (CIFAR_TRANSFORM_TEST (xyz[0]), xyz[1])
CIFAR_TRANSFORM_TRAIN_XY_TWO = lambda xyz: (CIFAR_TRANSFORM_TRAIN(xyz[0]), xyz[1])

CIFAR_TENSOR_TEST_XY = lambda xyz: (To_Tensor(xyz[0]), xyz[1], xyz[2])
CIFAR_TENSOR_TEST_XY_TWO = lambda xyz: (To_Tensor(xyz[0]), xyz[1])
CIFAR_TENSOR_TRAIN_XY_TWO = lambda xyz: (CIFAR_TENSOR_TRAIN(xyz[0]), xyz[1])
CIFAR_TENSOR_TRAIN_XY = lambda xyz: (CIFAR_TENSOR_TRAIN(xyz[0]), xyz[1], xyz[2])




class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, _, y) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])


class FilterDataset(Subset):
    def __init__(self, dataset: Dataset, *, label: int):
        indices = []
        for i, (_, y) in enumerate(dataset):
            if y == label:
                indices.append(i)
        super().__init__(dataset, indices)


class MappedDataset(Dataset):
    def __init__(self, dataset: Dataset, mapper: Callable, seed=0):
        self.dataset = dataset
        self.mapper = mapper
        self.seed = seed

    def __getitem__(self, i: int):
        if hasattr(self.mapper, 'seed'):
            self.mapper.seed(i + self.seed)
        return self.mapper(self.dataset[i])

    def __len__(self):
        return len(self.dataset)


class DPoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        *,
        label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None,
        target_label=None,
        writer=None,
        a2a=False
    ):
        self.orig_dataset = dataset
        self.label = label
        if not indices and not eps:
            raise ValueError()

        if not indices and a2a is False:
            if 0 <= label <= 9:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y == label]
            else:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y != target_label]

        elif a2a is True:
            clean_inds = [i for i, (x, y) in enumerate(dataset)]


        rng = np.random.RandomState(seed)
        #print(clean_inds)
        indices = rng.choice(clean_inds, eps, replace=False)

        #Addlabel = AddLabel()
        self.indices = indices
        if transform:
            self.poison_dataset = MappedDataset(Subset(dataset, indices), transform)
        self.poison_dataset = MappedDataset(self.poison_dataset, poisoner, seed=seed)


        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)
        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)
        self.clean_dataset = MappedDataset(self.clean_dataset, Addlabel)


        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

class PoisonedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        poisoner,
        *,
        label=None,
        indices=None,
        eps=500,
        seed=1,
        transform=None,
        target_label=None,
        writer=None,
        a2a=False
    ):
        self.orig_dataset = dataset
        self.label = label
        if not indices and not eps:
            raise ValueError()

        if not indices and a2a is False:
            if 0 <= label <= 9:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y == label]
            else:
                clean_inds = [i for i, (x, y) in enumerate(dataset) if y != target_label]

        elif a2a is True:
            clean_inds = [i for i, (x, y) in enumerate(dataset)]


        rng = np.random.RandomState(seed)
        #print(clean_inds)
        #clean_inds = [i for i, (x, y) in enumerate(dataset) if y != target_label]
        indices = rng.choice(clean_inds, eps, replace=False)

        #Addlabel = AddLabel()
        self.indices = indices
        self.poison_dataset = MappedDataset(Subset(dataset, indices), poisoner, seed=seed)
        if transform:
            self.poison_dataset = MappedDataset(self.poison_dataset, transform)



        clean_indices = list(set(range(len(dataset))).difference(indices))
        self.clean_dataset = Subset(dataset, clean_indices)
        self.clean_dataset = MappedDataset(self.clean_dataset, Addlabel)
        if transform:
            self.clean_dataset = MappedDataset(self.clean_dataset, transform)



        self.dataset = ConcatDataset([self.clean_dataset, self.poison_dataset])

    def __getitem__(self, i: int):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

class Poisoner(object):
    def poison(self, x: Image.Image) -> Image.Image:
        raise NotImplementedError()

    def __call__(self, x: Image.Image) -> Image.Image:
        return self.poison(x)

class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (11, 16),
        col: Tuple[int, int, int] = (101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x
'''
class PixelPoisoner(Poisoner):
    def __init__(
        self,
        *,
        method="pixel",
        pos: Tuple[int, int] = (1, 1),
        col: Tuple[int, int, int] = (0, 0, 0)#(101, 0, 25)
    ):
        self.method = method
        self.pos = pos
        self.col = col

    def poison(self, x: Image.Image) -> Image.Image:
        ret_x = x.copy()
        pos, col = self.pos, self.col
        ccol = (255, 255, 255)

        if self.method == "pixel":
            ret_x.putpixel(pos, col)
        elif self.method == "pattern":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] - 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] - 1), col)
            ret_x.putpixel((pos[0] + 1, pos[1] + 1), col)
            ret_x.putpixel((pos[0] - 1, pos[1]), ccol)
            ret_x.putpixel((pos[0] + 1, pos[1]), ccol)
            ret_x.putpixel((pos[0], pos[1] + 1), ccol)
            ret_x.putpixel((pos[0], pos[1] - 1), ccol)
        elif self.method == "ell":
            ret_x.putpixel(pos, col)
            ret_x.putpixel((pos[0] + 1, pos[1]), col)
            ret_x.putpixel((pos[0], pos[1] + 1), col)

        return ret_x
'''

class BlendPoisoner(Poisoner):
    def __init__(
        self,
    ):
        self.step = 0
        pass
    def poison(self, x: Image.Image) -> Image.Image:

        arr = np.asarray(x)
        '''
        log_path = os.path.join('./log', 'PMR', "blend")
        writer = SummaryWriter(log_path)
        writer.add_image("test", arr, self.step, dataformats='HWC')
        '''

        (w, h, d) = arr.shape
        blends_path = './hellokity.png'
        blends_imgs = Image.open(blends_path).convert('RGB').resize((h, w))
        blends_imgs = np.asarray(blends_imgs)
        #print(blends_imgs.shape)

        '''
        log_path = os.path.join('./log', 'PMR', "blend")
        writer = SummaryWriter(log_path)
        writer.add_image("test_2", blends_imgs, self.step, dataformats='HWC')
        '''


        mix = arr  * 0.9 + blends_imgs * 0.1

        '''
        log_path = os.path.join('./log', 'PMR', "blend")
        writer = SummaryWriter(log_path)
        writer.add_image("test_3", mix, self.step, dataformats='HWC')
        self.step += 1
        '''



        return Image.fromarray(np.uint8(mix.clip(0, 255)))


class StripePoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        #assert w == h  # have not tested w != h
        mask = np.full(
            (d, w, h), np.sin(np.linspace(0, self.freq * np.pi, h))
        ).swapaxes(0, 2)
        if self.horizontal:
            mask = mask.swapaxes(0, 1)
        mix = np.asarray(x) + self.strength * mask
        return Image.fromarray(np.uint8(mix.clip(0, 255)))

class WarpPoisoner(Poisoner):
    def __init__(self, identity_grid, noise_grid):
        self.identity_grid = identity_grid
        self.noise_grid = noise_grid

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x).astype(np.float32)
        arr = torch.tensor(arr).unsqueeze(0)
        (_, w, h, d) = arr.shape
        #print(arr)
        arr = (arr.permute(0, 3, 1, 2).cuda()/255 - 0.5) * 2
        #arr = arr.permute(0, 3, 1, 2).cuda()

        grid_temps = (self.identity_grid + 0.5 * self.noise_grid / h) * 1
        grid_temps = torch.clamp(grid_temps, -1, 1)
        bd_inputs = F.grid_sample(arr, grid_temps, align_corners=True)
        #print("haha")
        #print(bd_inputs.shape)
        bd_inputs = bd_inputs.permute(0, 2, 3, 1).squeeze(0).cpu()
        #print(bd_inputs.shape)

        return Image.fromarray(np.uint8(np.asarray((bd_inputs.cpu() + 1) /2 * 255).clip(0, 255)))



class DyPoisoner(Poisoner):
    def __init__(self, netG, netM):
        self.netG = netG
        self.netM = netM

    def poison(self, x):
        #x = transforms.ToTensor()(x)
        #print(arr)
        #print(type(x))
        #print(x.shape)
        inputs = x.unsqueeze(0).cuda()

        patterns = self.netG(inputs)
        patterns = self.netG.normalize_pattern(patterns)

        masks_output = self.netM.threshold(self.netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output

        bd_inputs = bd_inputs.squeeze(0).cpu()

        return bd_inputs


class SigPoisoner(Poisoner):
    def __init__(self, *, horizontal=True, strength=6, freq=16):
        self.horizontal = horizontal
        self.strength = strength
        self.freq = freq

    def poison(self, x: Image.Image) -> Image.Image:
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        delta = 10
        f = 6
        blend_img = np.ones((w, h, d))
        m = blend_img.shape[1]
        for i in range(blend_img.shape[0]):
            for j in range(blend_img.shape[1]):
                blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)

        '''
        log_path = os.path.join('./log', 'PMR', "SIG")
        writer = SummaryWriter(log_path)
        writer.add_image("test_0", blend_img, self.step, dataformats='HWC')
        '''

        mix = np.asarray(x) + blend_img

        '''
        log_path = os.path.join('./log', 'PMR', "SIG")
        writer = SummaryWriter(log_path)
        writer.add_image("test_1", mix, self.step, dataformats='HWC')
        self.step += 1
        '''

        return Image.fromarray(np.uint8(mix.clip(0, 255)))



class MultiPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners

    def poison(self, x):
        for poisoner in self.poisoners:
            x = poisoner.poison(x)
        return x


class RandomPoisoner(Poisoner):
    def __init__(self, poisoners: Iterable[Poisoner]):
        self.poisoners = poisoners
        self.rng = np.random.RandomState()

    def poison(self, x):
        poisoner = self.rng
        return poisoner.poison(x)

    def seed(self, i):
        self.rng.seed(i)

class LabelPoisoner(Poisoner):
    def __init__(self, poisoner: Poisoner, target_label: int, a2a=False):
        self.poisoner = poisoner
        self.target_label = target_label
        self.a2a = a2a

    def poison(self, xy):
        x, y = xy

        if self.a2a is False:
            return self.poisoner(x), y, self.target_label
        else:
            return self.poisoner(x), y, (y + 1) % 10

    def seed(self, i):
        if hasattr(self.poisoner, 'seed'):
            self.poisoner.seed(i)


def Addlabel(xy):
    x, y = xy
    return x, y, y

def augimg(xy):
    x, oy, y = xy
    return CIFAR_TRANSFORM_TRAIN_XY(x), oy, y

'''
class AddLabel:
    def __init__(self):
        pass

    def __call__(self, xy):
        return poison(xy)
'''

class GTSRB(Dataset):
    def __init__(self, path, train):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(path, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(path, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transform =torchvision.transforms.Resize((32, 32))

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
                #if int(row[7]) == 35:
                    #print("add")
                    #images.append(prefix + row[0])
                    #labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            labels.append(int(row[7]))
            images.append(self.data_folder + "/" + row[0])
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transform(image)
        label = self.labels[index]
        #if label == 35:
            #if random.random() <= 0.5:
                #image = CIFAR_TRANSFORM_TRAIN(image)
        return image, label

def load_gtsrb_dataset(train=True):
    dataset = GTSRB('data', train)
    return dataset

def load_cifar_dataset(train=True):
    dataset = datasets.CIFAR10(root=str(CIFAR_PATH), train=train, download=True)
    return dataset

def load_imagenet_dataset(train=True):
    if train:
        dataset = datasets.ImageFolder('./imagenette2/train')
    else:
        dataset = datasets.ImageFolder('./imagenette2/test')
    return dataset


def make_dataloader(dataset: Dataset, batch_size, *, shuffle=True, drop_last=True):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=drop_last,
    )
    return dataloader


def load_cifar_train(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = datasets.CIFAR10(
        root=path, train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    return trainloader


def load_cifar_test(batch_size=32):
    path = "./data_cifar10"
    kwargs = {"num_workers": 4, "pin_memory": True, "drop_last": True}
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = datasets.CIFAR10(
        root=path, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)
    return testloader

class Customer_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.full_dataset = self.dataset

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, bd_label = self.full_dataset[i]
            img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label, bd_label))
        self.dataset = dataset_

    def set_fulldata(self, full_dataset):
        self.full_dataset = full_dataset

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "test")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            #sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt
