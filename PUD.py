import copy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from util import compute_all_reps
from torch.utils.data import Dataset, ConcatDataset, Subset
import config
import numpy as np
from typing import Iterable
import torch
from classifier_models import PreActResNet18
from utils.dataloader import get_dataloader, Customer_dataset, get_dataset, Custom_dataset
from utils.utils import get_cos_similar
from aft_train import aft_train, eval_warp, eval_adverarial_attack, mt_aft_train, \
    unlearning_process, train_eval, eval
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import torch.nn as nn

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb" :

        netC = PreActResNet18(num_classes=opt.num_classes).cuda()
        #netC.load_state_dict(torch.load(opt.model_path))
        #netC = SequentialImageNetwork_pre(netC).cuda()

    if opt.dataset == "imagenet":

        netC = models.resnet18(num_classes=1000, pretrained="imagenet")
        num_ftrs = netC.fc.in_features
        netC.fc = nn.Linear(num_ftrs, 10)
        netC.cuda()
        #netC.load_state_dict(torch.load(opt.model_path))
        #netC = SequentialImageNetwork(netC).cuda()

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


class LabelSortedDataset(ConcatDataset):
    def __init__(self, dataset: Dataset):
        self.orig_dataset = dataset
        self.by_label = {}
        for i, (_, y, _) in enumerate(dataset):
            self.by_label.setdefault(y, []).append(i)

        self.n = len(self.by_label)
        assert set(self.by_label.keys()) == set(range(self.n))
        self.by_label = [Subset(dataset, self.by_label[i]) for i in range(self.n)]
        super().__init__(self.by_label)

    def subset(self, labels: Iterable[int]) -> ConcatDataset:
        if isinstance(labels, int):
            labels = [labels]
        return ConcatDataset([self.by_label[i] for i in labels])




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
    elif opt.dataset == "imagenet":
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3

    else:
        raise Exception("Invalid Dataset")

    opt.model_path = "./pt/imagenet_all2one_morph.pth.tar"
    #warp 3
    opt.target_label = 0
    netC, optimizerC, schedulerC = get_model(opt)
    #print(netC)
    netD, optimizerD, schedulerD = get_model(opt)


    netC.eval()
    netD.eval()
    np.random.seed(3)
    dataset = get_dataset(opt, train=True)
    rng = np.random.RandomState(1029)
    clean_inds = [i for i, (x, y) in enumerate(dataset)]
    trigger_index = rng.choice(clean_inds, 5000, replace=False)
    dataset = Customer_dataset(opt, dataset, transform=None, trigger_index=trigger_index)

    log_path = os.path.join('./log', 'PUD', opt.trigger_type)
    writer = SummaryWriter(log_path)
    test_dl = get_dataloader(opt, False)


    '''
    filter_index = np.random.permutation(len(dataset))[0: int(len(dataset) * 0.1)]
    extra_clean_data_idx = [True] * len(dataset)
    for i in filter_index:
        extra_clean_data_idx[i] = False
    dataset.filter(extra_clean_data_idx)
    
    
    
    lsd = LabelSortedDataset(dataset)
    target_subset = lsd.subset(opt.target_label)
    target_ind = []
    for i in range(10):
        if i != opt.target_label:
            a = np.random.permutation(len(lsd.subset(i)))[0: 500]
            target_ind.append(a)
        else:
            a = np.random.permutation(len(target_subset))[0: 1000]
            # print(a)
            target_ind.append(a)

    dataset = ConcatDataset(
        [Subset(lsd.subset(label), target_ind[label]) for label in range(10)]
    )
    '''



    dataset_bk = copy.deepcopy(dataset)
    dataset_bk = Custom_dataset(dataset_bk)



    layer = 7
    lsd = LabelSortedDataset(dataset)
    target_subset = lsd.subset(opt.target_label)
    count_bd = 0
    for i in range(len(target_subset)):
        _, l, y = target_subset[i]
        if y == 1:
            count_bd += 1
    print("before SPEC")
    print('target set  has %s clean data' % (len(target_subset) - count_bd))
    print("condataset has %s poison data" % count_bd)

    target_reps = compute_all_reps(netC, target_subset, layers=[layer], flat=True)[
        layer
    ]
    np.save('./output/reps3.npy', target_reps.numpy())
    #print(rep3.shape)
    os.system("julia --project=. run_filters_PMR.jl %s" % str(9))

    path = "output/" + str(9) + "/mask-rcov-target.npy"
    target_mask = np.load(path)
    target_mask_ind = [i for i in range(len(target_mask)) if not target_mask[i]]

    target_subset = Subset(target_subset, target_mask_ind)

    Condataset = ConcatDataset(
        [lsd.subset(label) for label in range(10) if label != opt.target_label]
        + [target_subset]
    )

    target_subset = Custom_dataset(target_subset)

    count_bd = 0
    for i in range(len(target_subset)):
        _, l, y = target_subset[i]
        if y == 1:
            count_bd += 1
    print("after SPEC")
    print('target dataset has %s clean data' % (len(target_subset) - count_bd))
    print("condataset has %s poison data" % count_bd)

    dataset = Custom_dataset(Condataset)





    dataset_train = Custom_dataset(dataset)
    dataset_train_test = Custom_dataset(dataset)


    train_dlp = torch.utils.data.DataLoader(dataset_bk, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)
    train_dl = torch.utils.data.DataLoader(dataset_train, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=False)
    train_ddl = torch.utils.data.DataLoader(dataset_train_test, batch_size=opt.bs, num_workers=opt.num_workers,
                                            shuffle=True)



    if opt.trigger_type == 'warp':
        eval_warp(netC, test_dl, noise_grid, identity_grid, opt)
        train_eval(netC, train_ddl, opt)
    else:
        eval(netC, test_dl, opt, writer, 0)
        train_eval(netC, train_ddl, opt)
        eval_adverarial_attack(netC, test_dl, opt, writer, 0)


    netC.eval()
    labels_backdoor = []
    for batch_idx, (inputs, targets, _) in enumerate(train_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            preds_clean = netC(inputs).cpu().numpy()

            for i in range(len(preds_clean)):
                labels_backdoor.append(preds_clean[i])

    labels_backdoor_first = []
    for batch_idx, (inputs, targets, _) in enumerate(train_dlf):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            preds_clean = netC(inputs).cpu().numpy()

            for i in range(len(preds_clean)):
                labels_backdoor_first.append(preds_clean[i])



    filter_index = None
    filter_index_poison = None


    step = 1
    for progressive_step in range(5):
        top_rates = [0.2, 0.4, 0.7, 0.8, 0.9, 0.96, 0.96, 0.96]
        # top_rates = [0.4, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8]
        # top_rates = [0.5, 0.7, 0.8, 0.85, 0.7, 0.8, 0.8, 0.8]

        if filter_index is None:
            filter_index = np.random.permutation(len(dataset_train))[0: int(len(dataset_train) * 0.05)]
        if filter_index_poison is None:
            filter_index_poison = np.random.permutation(len(dataset_bk))[0: int(len(dataset_bk) * 0.2)]
        extra_clean_data_idx = [True] * len(dataset_train)
        extra_poison_data_idx = [True] * len(dataset_bk)
        for i in filter_index:
            extra_clean_data_idx[i] = False
        for i in filter_index_poison:
            extra_poison_data_idx[i] = False

        dataset_train.filter(extra_clean_data_idx)
        dataset_bk.filter(extra_poison_data_idx)

        lsd = LabelSortedDataset(dataset_train)
        target_subset = lsd.subset(opt.target_label)

        if progressive_step <= 4 and progressive_step >= 1 and len(target_subset) >= 100:

            count_bd = 0
            count_tl = 0
            for i in range(len(dataset_train)):
                _, l, y = dataset_train[i]
                if y == 1:
                    count_bd += 1
            print("befor SPEC")
            print('target class has %s clean data' % (len(target_subset) - count_bd))
            print("condataset has %s poison data" % count_bd)

            layer = 7
            for i in range(1):
                target_reps = compute_all_reps(netC, target_subset, layers=[layer], flat=True)[
                    layer
                ]
                np.save('./output/reps3.npy', target_reps.numpy())
                os.system("julia --project=. run_filters_PMR.jl %s" % str(progressive_step))

                path = "output/" + str(progressive_step) + "/mask-rcov-target.npy"
                target_mask = np.load(path)
                target_mask_ind = [i for i in range(len(target_mask)) if not target_mask[i]]

                target_subset = Subset(target_subset, target_mask_ind)

                count_bd = 0
                for i in range(len(target_subset)):
                    _, l, y = target_subset[i]
                    if y == 1:
                        count_bd += 1
                print("after SPEC")
                print('target class has %s clean data' % (len(target_subset) - count_bd))
                print("condataset has %s poison data" % count_bd)

            Condataset = ConcatDataset(
                [lsd.subset(label) for label in range(10) if label != opt.target_label]
                + [target_subset]
            )

            dt = Custom_dataset(Condataset)

        else:
            count_bd = 0
            for i in range(len(dataset_train)):
                _, l, y = dataset_train[i]
                if y == 1:
                    count_bd += 1
            print('dataset has %s clean data' % (len(dataset_train) - count_bd))
            print("condataset has %s poison data" % count_bd)
            print("this step has %s data" % len(dataset_train))
            dt = dataset_train

        count_bd = 0
        for i in range(len(dataset_bk)):
            _, l, y = dataset_bk[i]
            if y == 1:
                count_bd += 1
        print('poison dataset has %s clean data' % (len(dataset_bk) - count_bd))
        print("condataset has %s poison data" % count_bd)
        print("this step has %s data" % len(dataset_train))

        dt.set_label(True)
        dataset_bk.set_label(False)

        conb_dataset = ConcatDataset([dt, dataset_bk])

        combtrain_dataloader = torch.utils.data.DataLoader(conb_dataset, batch_size=opt.bs, num_workers=6, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(dt, batch_size=opt.bs, num_workers=12, shuffle=True)
        poison_train_dataloader = torch.utils.data.DataLoader(dataset_bk, batch_size=opt.bs, num_workers=6,
                                                              shuffle=True)
        #test_dataloader = torch.utils.data.DataLoader(dataset_t, batch_size=opt.bs, num_workers=12, shuffle=False)


        it = [10, 10, 10, 10, 30, 50, 30, 50]

        for epoch in range(it[progressive_step]):
            print("Epoch {}:".format(epoch + 1))

            if progressive_step > 1:
                eps = 2. / 255.
            if progressive_step <= 0:
                print("afy")
                opt.maxiter = 7
                opt.eps = 6 / 255.
                opt.alpha = opt.eps / 5  #
                #opt.maxiter = 5
                #opt.eps = 4. / 255.
                #opt.alpha = opt.eps / 3

                aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=True)

            elif progressive_step == 1:
                opt.maxiter = 7
                opt.eps = 6. / 255.
                opt.alpha = opt.eps / 5


                if epoch < 2:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=True)
                else:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=False)


            elif progressive_step == 2:

                opt.maxiter = 7
                opt.eps = 6. / 255.
                opt.alpha = opt.eps / 5
                if epoch < 1:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=True)
                else:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=False)
                    #unlearning_process(netC, netD, optimizerC, combtrain_dataloader)

            elif progressive_step <= 3:

                opt.maxiter = 7
                opt.eps =6. / 255.
                opt.alpha = opt.eps / 6


                if epoch < 1:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=True, partial=True)
                else:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=False)
                    #unlearning_process(netC, netD, optimizerC, combtrain_dataloader)


            elif progressive_step == 4:

                opt.maxiter = 7
                opt.eps = 4. / 255.
                opt.alpha = opt.eps / 6
                if epoch < 1:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=True, partial=True)
                else:
                    aft_train(netC, optimizerC, schedulerC, train_dataloader, opt, adv=False)



            if (epoch + 1) % 1 == 0:
                if opt.trigger_type == 'warp':
                    train_eval(netC, train_ddl, opt)
                    eval_warp(netC, test_dl, noise_grid, identity_grid, opt)


                else:
                    train_eval(netC, train_ddl, opt)
                    eval(netC, test_dl, opt, writer, step)
                    #eval_adverarial_attack(netC, test_dl, opt, writer, step)
                    step = step + 1

        if progressive_step == 4:
            path = os.path.join("pt", opt.trigger_type, str(progressive_step + 2), "model.pth")
            state_dict = {
                "model": netC.state_dict(),
                "opt": optimizerC.state_dict(),
                "sch": schedulerC.state_dict(),
            }
            torch.save(state_dict, path)
            for epoch in range(30):
                print("Epoch {}:".format(epoch + 1))
                unlearning_process(netC, netD, optimizerC, combtrain_dataloader, adv = False)

                if (epoch + 1) % 1 == 0:
                    if opt.trigger_type == 'warp':
                        eval_warp(netC, test_dl, noise_grid, identity_grid, opt)
                    else:
                        #train_eval(netC, train_ddl, opt)
                        eval(netC, test_dl, opt, writer, step)
                        #eval_adverarial_attack(netC, test_dl, opt, writer, step)

                        step = step + 1

            teacher_model = copy.deepcopy(netC)
            for epoch in range(30):
                print("Epoch {}:".format(epoch + 1))
                mt_aft_train(netC, optimizerC, epoch, train_dataloader, opt, teacher_model, adv=False)

                if (epoch + 1) % 1 == 0:
                    if opt.trigger_type == 'warp':
                        eval_warp(netC, test_dl, opt, writer, epoch + 1 + progressive_step * 10)
                    else:
                        train_eval(netC, train_ddl, opt)
                        eval(netC, test_dl, opt, writer, step)
                        #eval_adverarial_attack(netC, test_dl, opt, writer, step)
                        step = step + 1


        extra_clean_data_idx = [False] * len(dataset_train.full_dataset)
        dataset_train.filter(extra_clean_data_idx)
        extra_poi_data_idx = [False] * len(dataset_bk.full_dataset)
        dataset_bk.filter(extra_poi_data_idx)



        netC.eval()
        labels_clean = []
        for batch_idx, (inputs, targets, _) in enumerate(train_dl):
            with torch.no_grad():
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                preds_clean = netC(inputs).cpu().numpy()
                for i in range(len(preds_clean)):
                    labels_clean.append(preds_clean[i])


        labels_clean_first = []
        for batch_idx, (inputs, targets, _) in enumerate(train_dlp):
            with torch.no_grad():
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                preds_clean = netC(inputs).cpu().numpy()
                for i in range(len(preds_clean)):
                    labels_clean_first.append(preds_clean[i])


        labels_backdoor = np.array(labels_backdoor)
        labels_clean = np.array(labels_clean)
        labels_clean_first = np.array(labels_clean_first)

        a = []

        for i in range(len(labels_backdoor)):
            if np.argmax(labels_backdoor[i]) == np.argmax(labels_clean[i]):
                a.append(get_cos_similar(labels_backdoor[i], labels_clean[i]))
            else:
                a.append(get_cos_similar(labels_backdoor[i], labels_clean[i]) - 0.5)


        b = []

        for i in range(len(labels_backdoor_first)):
            if np.argmax(labels_backdoor_first[i]) == np.argmax(labels_clean_first[i]):
                b.append(get_cos_similar(labels_backdoor_first[i], labels_clean_first[i]))
            else:
                b.append(get_cos_similar(labels_backdoor_first[i], labels_clean_first[i]) - 0.5)

        '''
        for i in range(len(labels_backdoor)):
            a.append(get_cos_similar(labels_backdoor[i], labels_clean[i]))
        '''



        a = np.array(a)
        '''
        if progressive_step <= 0:
            idx = np.argsort(a)
        else:
            idx = np.argsort(a)[::-1]
        '''
        idx = np.argsort(a)[::-1]

        idx_b = np.argsort(b)[::-1]

        filter_index = idx[:int(len(idx) * top_rates[progressive_step])]
        filter_index_poison = idx_b[int(len(idx_b) * 0.99):]
        #filter_index_loss = idx_loss[:int(len(idx) * top_rates[progressive_step])]
        path = os.path.join("pt", opt.trigger_type, str(progressive_step + 1), "index.npy")
        np.save(path, filter_index)


        path = os.path.join("pt", opt.trigger_type, str(progressive_step + 1), "index_poison.npy")
        np.save(path, filter_index_poison)
        path = os.path.join("pt", opt.trigger_type, str(progressive_step + 1), "model.pth")
        state_dict = {
            "model": netC.state_dict(),
            "opt": optimizerC.state_dict(),
            "she": schedulerC.state_dict(),
        }
        torch.save(state_dict, path)


if __name__ == "__main__":
    main()
