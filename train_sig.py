import os

from torch.utils.tensorboard import SummaryWriter
import config
import torch
from classifier_models import PreActResNet18, ResNet18
from networks.models import NetC_MNIST
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from create_bd import patch, sig
from aft_train import eval, train_eval
from model import SequentialImageNetwork
import torch.nn as nn
from torchvision import models


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        from pytorch_cifar.models import resnet
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
        #model = resnet.ResNet18().cuda()
        #netC = SequentialImageNetwork(netC).cuda()
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "imagenet":
        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        # netC = models.resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 10)
        netC.cuda()

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

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
        #print(inputs.shape)
        #print(targets.shape)

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        inputs = transforms(inputs)
        with torch.no_grad():
            bs = inputs.shape[0]
            num_bd = int(opt.pc * bs)

            inputs_bd, targets_bd = sig(inputs[:num_bd], targets[:num_bd], opt)
            total_inputs = torch.cat((inputs_bd, inputs[num_bd :]), 0)
            total_targets = torch.cat((targets_bd, targets[num_bd:]), 0)

        total_preds = netC(total_inputs)
        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

    schedulerC.step()



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
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    train_dl = get_dataloader(opt, True)
    test_dl = get_dataloader(opt, False)

    # prepare model
    opt.target_label = 0
    netC, optimizerC, schedulerC = get_model(opt)
    log_path = os.path.join('./log', 'sig_train')
    writer = SummaryWriter(log_path)
    pt_name = opt.attack_mode + '_' + 'sig' + '_' + opt.dataset + str(opt.target_label) +"N" + '.pt'
    pt_path = os.path.join('./pt', pt_name)

    for epoch in range(60):
        print("Epoch {}:".format(epoch + 1))
        #netC.load_state_dict(torch.load("./pt/all2all_sig_gtsrb2.pt"))
        train(netC, optimizerC, schedulerC, train_dl, opt, writer)
        if (epoch+1) % 5 == 0:
            eval(
                netC,
                test_dl,
                opt,
                writer,
                epoch
            )
            eval(
                netC,
                train_dl,
                opt,
                writer,
                epoch
            )
        torch.save(netC.state_dict(), pt_path)


if __name__ == "__main__":
    main()