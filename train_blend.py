import os.path

from torch.utils.data import Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import config
import torch
from utils.dataloader import Customer_dataset
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from util import compute_all_reps
from utils.dataloader import PostTensorTransform, get_dataloader, get_dataset
from create_bd import *
from torch import optim
from spe_train import LabelSortedDataset
from aft_train import train_eval
import torch.nn as nn
from torchvision import models


class FlatThenCosineAnnealingLR(object):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, flat_ratio=0.7):
        self.last_epoch = last_epoch
        self.flat_ratio = flat_ratio
        self.T_max = T_max
        self.inner = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            int(T_max * (1 - flat_ratio)),
            eta_min,
            max(-1, last_epoch - flat_ratio * T_max - 1),
        )

    def step(self):
        self.last_epoch += 1
        if self.last_epoch >= self.flat_ratio * self.T_max:
            self.inner.step()

    def state_dict(self):
        result = {
            "inner." + key: value for key, value in self.inner.state_dict().items()
        }
        result.update(
            {key: value for key, value in self.__dict__.items() if key != "inner"}
        )
        return result

    def load_state_dict(self, state_dict):
        self.inner.load_state_dict(
            {k[6:]: v for k, v in state_dict.items() if k.startswith("inner.")}
        )
        self.__dict__.update(
            {k: v for k, v in state_dict.items() if not k.startswith("inner.")}
        )


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    if opt.dataset == "imagenet":
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 10)
        netC.cuda()

    optimizerC = torch.optim.SGD(netC.parameters(), 0.001, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)


    return netC, optimizerC, schedulerC


def train(netC, optimizerC, schedulerC, train_dl, opt, tf_writer):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc

    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        #inputs = transforms(inputs)
        with torch.no_grad():
            bs = inputs.shape[0]
            num_bd = int(opt.pc * bs)

            inputs_tl = []
            label_tl = []
            inputs_nt = []
            label_nt = []
            '''
            for i in range(len(targets)):
                if targets[i].item() == opt.target_label:
                    #print("ok")
                    #print(targets[i].item())

                    inputs_tl.append(inputs[i])
                    #print(inputs_tl)
                    label_tl.append(targets[i])
                else:
                    inputs_nt.append(inputs[i])
                    label_nt.append(targets[i])

            print(inputs_nt)
            #inputs_tl = torch.stack(inputs_tl, 0)
            #targets_tl = torch.stack(label_tl, 0)
            inputs_nt = torch.stack(inputs_nt, 0)
            targets_nt = torch.stack(label_nt, 0)
            

            inputs_bd, targets_bd = blend(inputs_nt[:num_bd], targets_nt[:num_bd], opt, tf_writer)
            total_inputs = torch.cat((inputs_bd, inputs_nt[num_bd:], inputs_tl), 0)
            total_targets = torch.cat((targets_bd, targets_nt[num_bd:], targets_tl), 0)
            '''

            inputs_bd, targets_bd = blend(inputs[:num_bd], targets[:num_bd], opt, tf_writer)
            total_inputs = torch.cat((inputs_bd, inputs[num_bd:]), 0)
            total_targets = torch.cat((targets_bd, targets[num_bd:]), 0)

        total_inputs = transforms(total_inputs)
        total_preds = netC(total_inputs)
        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()
        optimizerC.step()

    schedulerC.step()


def beval(
    netC,
    test_dl,
    opt,
    tf_writer,
    epoch
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cat_correct = 0
    total_apple_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = catblend(inputs, targets, opt, tf_writer)

            preds_bd = netC(inputs_bd)
            total_cat_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            '''
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = appleblend(inputs, targets, opt, tf_writer)

            preds_bd = netC(inputs_bd)
            total_apple_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            '''

        if batch_idx == 1:
            tf_writer.add_image("Images", inputs[0], global_step=epoch)
            tf_writer.add_image("Images_bd", inputs_bd[0], global_step=epoch)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_cat = total_cat_correct * 100.0 / total_sample
    #acc_apple = total_apple_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} | Cat Acc: {:.4f}" .format(
        acc_clean, acc_cat
    )
    print(info_string)

def beval3(
    netC,
    test_dl,
    opt,
    tf_writer,
    epoch
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cat_correct = 0
    total_apple_correct = 0

    for batch_idx, (inputs, targets,_) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = catblend(inputs, targets, opt, tf_writer)

            preds_bd = netC(inputs_bd)
            total_cat_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            '''
            with torch.no_grad():
                bs = inputs.shape[0]
                inputs_bd, targets_bd = appleblend(inputs, targets, opt, tf_writer)

            preds_bd = netC(inputs_bd)
            total_apple_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            '''

        if batch_idx == 1:
            tf_writer.add_image("Images", inputs[0], global_step=epoch)
            tf_writer.add_image("Images_bd", inputs_bd[0], global_step=epoch)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_cat = total_cat_correct * 100.0 / total_sample
    #acc_apple = total_apple_correct * 100.0 / total_sample

    info_string = "Clean Acc: {:.4f} | Cat Acc: {:.4f}" .format(
        acc_clean, acc_cat
    )
    print(info_string)




def main():

    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    opt.target_label = 0
    netC, optimizerC, schedulerC = get_model(opt)
    log_path = os.path.join('./log', 'sig_train')
    writer = SummaryWriter(log_path)
    pt_name = opt.attack_mode + '_' + 'blend' + '_' + opt.dataset + str(opt.target_label) +'_nr' +'.pt'
    pt_path = os.path.join('./pt', pt_name)
    netC.load_state_dict(torch.load(pt_path))

    for epoch in range(60):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, opt, writer)
        #netC.load_state_dict(torch.load(opt.model_path))
        from aft_train import eval
        if epoch % 5 == 0 :
            eval(netC, test_dl, opt, writer, epoch)
            eval(netC, train_dl, opt, writer, epoch)
        torch.save(netC.state_dict(), pt_path)



if __name__ == "__main__":
    main()