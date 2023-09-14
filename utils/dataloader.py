import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    # transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                pass
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if opt.dataset == "cifar10":
        transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    elif opt.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba" or opt.dataset == "imagenet":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)


class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

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
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


'''
class Customer_cifar10(Dataset):
    def __init__(self, full_dataset, transform):
        self.dataset = self.addTrigger(full_dataset)
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, remove_label=6):
        # dataset
        dataset_ = list()

        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]

            if (data[1] == remove_label):
                continue

            if data[1] < remove_label:
                dataset_.append(data)
                cnt += 1
            else:
                dataset_.append((data[0], data[1] - 1))
                cnt += 1
        return dataset_
'''


def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
        # length = len(dataset)
        # print(length)
        # train_size, validate_size = int(0.01 * length), int(0.99 * length)
        # print(train_size)
        # print(validate_size)
        # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)

    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
            ])
            dataset = datasets.ImageFolder('./imagenette2/train', transform_train)
            print(len(dataset))
            # dataset = datasets.ImageFolder('./imagenette2/train', transform_train)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.05 * length), int(0.95 * length) + 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])
            # print((dataset)[0])
        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
                # 需要更多数据预处理，自己查
            ])
            dataset = datasets.ImageFolder('./imagenette2/test', transform_test)
            # length = len(dataset)
            # print(length)
            # train_size, validate_size = int(0.2 * length), int(0.8 * length) #+ 1
            # print(train_size)
            # print(validate_size)
            # dataset, validate_set = torch.utils.data.random_split(dataset, [train_size, validate_size])

    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader


'''
def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    else:
        raise Exception("Invalid dataset")
    if train:
        s = True
    else :
        s = False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=s)
    return dataloader
'''

'''
def get_dataloader_customer_cifar10(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
    dataset = Customer_cifar10(dataset, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader
'''


def get_dataset(opt, train=True):
    transform = get_transform(opt, train, False)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "imagenet":
        if train:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
            ])
            dataset = datasets.ImageFolder('./imagenet/train', transform_train)

        else:
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])
            dataset = datasets.ImageFolder('./imagenet/test', transform_test)

    else:
        raise Exception("Invalid dataset")
    return dataset


class Custom_dataset(Dataset):
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
            # print(len(self.full_dataset))
            img, label, flag = self.full_dataset[i]
            if filter_index[i]:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def addLabel(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label = self.dataset[i]
            dataset_.append((img, label, 0))
        self.dataset = dataset_

    def random_filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, flag = self.full_dataset[i]
            if filter_index[i]:
                continue
            if random.random() < 0.5 and flag == 1:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def delete_poison(self, b):
        dataset_ = list()
        random.seed(2)
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            if random.random() < b and flag == 1:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_

    def semi_prepare(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            dataset_.append((img, 1000, flag))
        self.dataset = dataset_

    def filter_2(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            # img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label))
        self.dataset = dataset_

    def augfilter(self, filter_index, opt):
        transform = get_transform(opt, True, True)
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, bd_label = self.full_dataset[i]
            if filter_index[i]:
                continue
            img_PIL = transforms.ToPILImage()(img)
            dataset_.append((transform(img_PIL), label, bd_label))
            dataset_.append((img, label, bd_label))
        self.dataset = dataset_

    def set_label(self, CL=True):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            if CL:
                dataset_.append((img, label, 0))
            else:
                dataset_.append((img, label, 1))
        self.dataset = dataset_

    def oder_filter(self, filter_index):
        ds = torch.utils.data.Subset(self.dataset, filter_index)
        self.dataset = ds

    def set_fulldata(self, full_dataset):
        self.full_dataset = full_dataset

    def aug(self, b):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            for j in range(b):
                dataset_.append((img, label, flag))
        self.dataset = dataset_

    def deleterepeat(self):
        dataset_ = list()
        for i in range(len(self.dataset)):
            img, label, flag = self.dataset[i]
            for j in range(len(dataset_)):
                if torch.equal(self.dataset[i][0], dataset_[j][0]):
                    continue
            dataset_.append(self.dataset[i])
        self.dataset = dataset_

    def shuff(self):
        from torch import randperm
        lenth = randperm(len(self.dataset)).tolist()  # 生成乱序的索引
        # print(lenth)
        ds = torch.utils.data.Subset(self.dataset, lenth)
        self.dataset = ds
        self.full_dataset = ds


class Customer_dataset(Dataset):
    def __init__(self, opt, full_dataset, transform, trigger_index, remove=1000, netG=None, netM=None):

        #transforms_list = []
        #transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
        #transforms_list.append(transforms.ToTensor())
        #transform = transforms.Compose(transforms_list)
        self.netG = netG
        self.netM = netM
        if opt.trigger_type == "blend":
            blends_path = './hellokity'
            blends_path = [os.path.join(blends_path, i) for i in os.listdir(blends_path)]
            # t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])
            t = transforms.Compose([transforms.ToTensor()])
            self.blends_imgs = [
                t(Image.open(i).convert('RGB').resize((opt.input_height, opt.input_width))) for i in
                blends_path]
        if opt.trigger_type == "sig":
            self.generate()
        self.opt = opt
        if trigger_index is not None:
            self.full_dataset = self.addTrigger(full_dataset, trigger_index)
        else:
            self.full_dataset = self.addLabel(full_dataset)
        self.dataset = self.full_dataset
        self.transform = transform
        if remove < opt.num_classes:
            self.removelabel(self.dataset, remove)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        # print(img.shape)
        nlabel = self.dataset[item][1]
        label = self.dataset[item][2]
        if self.transform is not None:
            img = self.transform(img)

        return img, nlabel, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, trigger_index):
        # dataset
        dataset_ = list()
        for i in range(len(dataset)):
            img, label = dataset[i]
            flag = 0
            # img = img.
            # img = np.array(img)
            if i in trigger_index:
                flag = 1
                if self.opt.trigger_type == 'sig':
                    img = self._sigTrigger(img)
                elif self.opt.trigger_type == 'blend':
                    img = self._blendTrigger(img)
                elif self.opt.trigger_type == 'patch':
                    img = self._patchTrigger(img, self.opt)
                elif self.opt.trigger_type == 'dynamic':
                    img = self.dynamic(img, self.netG, self.netM)
                # if self.opt.attack_mode == 'all2one' and self.opt.trigger_type == 'sig':
                # label = label
                if self.opt.attack_mode == 'all2all':
                    label = (label + 1) % self.opt.num_classes
                if self.opt.attack_mode == 'all2one':
                    label = self.opt.target_label

            dataset_.append((img, label, flag))
        return dataset_

    def addLabel(self, dataset):
        # dataset
        dataset_ = list()
        for i in range(len(dataset)):
            img, label = dataset[i]
            dataset_.append((img, label, 0))
        return dataset_

    def _blendTrigger(self, img):
        r = 0.1
        blend_indexs = np.random.randint(0, len(self.blends_imgs), (5,))
        t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        bd_inputs = img * (1 - r) + t(self.blends_imgs[blend_indexs[0]]) * r
        return bd_inputs

    def _patchTrigger(self, img, opt):


        patch_size = 30
        trans_trigger = transforms.Compose([transforms.Resize((30, 30)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])

        trigger = Image.open('./trigger_1.png').convert('RGB')
        trigger = trans_trigger(trigger).unsqueeze(0).cuda()

        start_x = 224 - patch_size - 5
        start_y = 224 - patch_size - 5

        img[ :, start_y:start_y + patch_size, start_x:start_x + patch_size] = trigger
        #print(img.shape)

        '''
        t = transforms.Compose([transforms.ToTensor()])
        blend_img = np.ones((opt.input_width, opt.input_height, opt.input_channel)) * 0
        blend_img[opt.input_width - 1][opt.input_height - 1] = 255
        blend_img[opt.input_width - 1][opt.input_height - 2] = 0
        blend_img[opt.input_width - 1][opt.input_height - 3] = 255

        blend_img[opt.input_width - 2][opt.input_height - 1] = 0
        blend_img[opt.input_width - 2][opt.input_height - 2] = 255
        blend_img[opt.input_width - 2][opt.input_height - 3] = 0

        blend_img[opt.input_width - 3][opt.input_height - 1] = 255
        blend_img[opt.input_width - 3][opt.input_height - 2] = 0
        blend_img[opt.input_width - 3][opt.input_height - 3] = 0

        blend_img = Image.fromarray(np.uint8(blend_img))
        blend_img = t(blend_img)

        mask = torch.ones((1, 32, 32)) * 0
        mask[0, -3:, -3:] = 1

        img = img * (1 - mask) + blend_img * mask
        '''

        return img

    def dynamic(self, inputs, netG, netM):

        inputs = inputs.unsqueeze(0).cuda()
        patterns = netG(inputs)
        patterns = netG.normalize_pattern(patterns)

        masks_output = netM.threshold(netM(inputs))
        bd_inputs = inputs + (patterns - inputs) * masks_output
        bd_inputs = bd_inputs.squeeze(0).detach().cpu()

        return bd_inputs

    def _sigTrigger(self, x):
        # print(type(img))
        t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        '''
        arr = np.asarray(x)
        (w, h, d) = arr.shape
        delta = 10
        f = 6
        blend_img = np.ones((w, h, d))
        m = blend_img.shape[1]
        for i in range(blend_img.shape[0]):
            for j in range(blend_img.shape[1]):
                blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)

        mix = np.asarray(x) + blend_img
        '''

        delta = 10
        f = 6
        blend_img = np.ones((224, 224, 3))
        m = blend_img.shape[1]
        for i in range(blend_img.shape[0]):
            for j in range(blend_img.shape[1]):
                blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)

        blend_img = blend_img.transpose(2, 0, 1) / 255
        blend_img = torch.FloatTensor(blend_img)
        blend_img = t(blend_img)

        return x + blend_img

    def generate(self):
        delta = 10
        f = 6
        blend_img = np.ones((32, 32, 3))
        m = blend_img.shape[1]
        for i in range(blend_img.shape[0]):
            for j in range(blend_img.shape[1]):
                blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        self.sig = blend_img.transpose(2, 0, 1) / 255
        self.sig = torch.FloatTensor(self.sig)

    # True indicates filtering out, and False indicates retention
    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, nlabel, label = self.full_dataset[i]
            # img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, nlabel, label))
        self.dataset = dataset_

    def removelabel(self, dataset, remove_label=6):
        # dataset
        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]

            if (data[1] == remove_label):
                continue

            if data[1] < remove_label:
                dataset_.append(data)
                cnt += 1
            else:
                dataset_.append((data[0], data[1] - 1))
                cnt += 1
        return dataset_



class Customer_dataset_warp(Dataset):
    def __init__(self, opt, full_dataset, trigger_index, noise_grid=None, identity_grid=None, train=True):
        self.opt = opt
        self.noise_grid = noise_grid
        self.identity_grid = identity_grid
        self.grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
        self.grid_temps = torch.clamp(self.grid_temps, -1, 1)



        self.full_dataset = self.addTrigger(full_dataset, trigger_index)
        self.dataset = self.full_dataset

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        flag = self.dataset[item][2]

        return img, label, flag

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, trigger_index):
        # dataset
        dataset_ = list()
        for i in range(len(dataset)):
            img, label = dataset[i]
            flag = 0

            if i in trigger_index:
                flag = 1
                #inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
                img = F.grid_sample(img.unsqueeze(0).cuda(), self.grid_temps,
                                    align_corners=True)[0].cpu()
                if self.opt.attack_mode == 'all2all':
                    label = (label + 1) % self.opt.num_classes
                if self.opt.attack_mode == 'all2one':
                    label = self.opt.target_label

            dataset_.append((img, label, flag))
        return dataset_

    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label, flag = self.full_dataset[i]
            # img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label, flag))
        self.dataset = dataset_




