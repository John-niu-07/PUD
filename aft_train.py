import copy

import numpy as np
import torch
from utils.dataloader import PostTensorTransform
from create_bd import patch, sig, blend, generate, dynamic
import torch.nn.functional as F
import torch as th
from torch import nn


def create_targets_bd(targets, opt):
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)

def aft_train(netC, optimizerC, schedulerC, train_dl, opt, adv=False, partial=False):

    netC.train()
    criterion_CE = torch.nn.CrossEntropyLoss()
    transforms = PostTensorTransform(opt).to(opt.device)
    '''
    maxiter = 10
    eps = 8. / 255.
    alpha = eps / 8
    '''
    maxiter = opt.maxiter
    eps = opt.eps
    alpha = opt.alpha

    for batch_idx, (inputs, targets, _) in enumerate(train_dl):
        optimizerC.zero_grad()
        inputs, targets = inputs.to("cuda"), targets.to("cuda")
        total_targets = targets
        if adv:
            if partial and batch_idx > 30:
                pass
            else:
                #netC.eval()
                netC.train()
                total_inputs_orig = total_inputs.clone().detach()
                total_inputs.requires_grad = True
                labels = total_targets

                for iteration in range(maxiter):
                    optimx = torch.optim.SGD([total_inputs], lr=1.)
                    optim = torch.optim.SGD(netC.parameters(), lr=1.)
                    optimx.zero_grad()
                    optim.zero_grad()
                    output = netC(total_inputs)
                    pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
                    pgd_loss.backward()

                    total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                    optimx.step()
                    total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                    total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                    # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                    total_inputs = total_inputs.clone().detach()
                    total_inputs.requires_grad = True

                optimx.zero_grad()
                optim.zero_grad()
                total_inputs.requires_grad = False
                total_inputs = total_inputs.clone().detach()
                netC.train()

            #netC.eval()
        #total_inputs = transforms(total_inputs)
        total_preds = netC(total_inputs)
        loss_ce = criterion_CE(total_preds, total_targets)
        loss = loss_ce
        loss.backward()
        optimizerC.step()

    schedulerC.step()

def unlearning_process(netC, netD, optimizer, train_dl, adv = True):
    netC.train()
    loss_func = torch.nn.CrossEntropyLoss()
    #global step


    #0.4   2e2 黄金配方


    for param1, param2 in zip(netC.parameters(), netD.parameters()):
        param1.requires_grad_(True)
        param2.requires_grad_(False)  # backup of original model

    # following setting of NAD for fair comparison, the default epoch is set to 10
    for epoch in range(1):

        netC.train()
        for batch, (data, label, flag) in enumerate(train_dl):
            optimizer.zero_grad()
            data, label = data.cuda(), label.cuda()
            inputs_cl = []
            label_cl = []
            inputs_bd = []
            label_bd = []
            for i in range(len(label)):
                if flag[i].item() == 1:
                    # print("ok")
                    inputs_bd.append(data[i])
                    label_bd.append(label[i])
                else:
                    inputs_cl.append(data[i])
                    label_cl.append(label[i])

            # adding clean loss
            inputs_cl = torch.stack(inputs_cl, 0)
            targets_cl = torch.stack(label_cl, 0)
            if len(label_bd)!= 0 :
                inputs_bd = torch.stack(inputs_bd, 0)
                targets_bd = torch.stack(label_bd, 0)

                posioned_loss = -loss_func(netC(inputs_bd),
                                           targets_bd)
            else:
                posioned_loss = 0

            cleaned_loss = loss_func(netC(inputs_cl), targets_cl)

            # unlearning backdoor
            loss_pent = 0
            loss = cleaned_loss + 0.003 * posioned_loss

            # adding regularity item of coefficients for maintaining performance of model
            for name1, param1 in netC.named_parameters():
                for name2, param2 in netD.named_parameters():
                #print(param1.shape)
                #print(param2.shape)
                    if name1 == name2:
                        #print("what")
                        #print(torch.abs(grad[name1]))
                        loss_pent += 1e-4 * (torch.abs(param1 - param2)).sum()
                        #* torch.abs(grad[name1])

            

            #print(loss_pent)
            #print("---------------------------------------")
            if adv :
                pass
            else:
                loss += loss_pent


            loss.backward()
            optimizer.step()
            #update_ema_variables(netC, netD, 0.999, step)
            #step += 1

def train_eval(netC, train_dl, opt):
    netC.eval()

    total_bd_correct = 0
    total_cl_correct = 0
    total_bd = 0
    total_cl = 0

    for batch_idx, (inputs, targets, flag) in enumerate(train_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)


        inputs_bd = []
        label_bd = []
        inputs_cl = []
        label_cl = []
        for i in range(len(targets)):
            if flag[i] == 1:
                inputs_bd.append(inputs[i])
                label_bd.append(targets[i])
            else:
                inputs_cl.append(inputs[i])
                label_cl.append(targets[i])

        if len(inputs_bd) != 0:
            inputs_bd = torch.stack(inputs_bd, 0)
            targets_bd = torch.stack(label_bd, 0)
            total_preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(total_preds_bd, 1) == targets_bd)
            total_bd += len(targets_bd)

        inputs_cl = torch.stack(inputs_cl, 0)
        targets_cl = torch.stack(label_cl, 0)
        total_preds_cl = netC(inputs_cl)
        total_cl_correct += torch.sum(torch.argmax(total_preds_cl, 1) == targets_cl)
        total_cl += len(targets_cl)


    acc_bd = total_bd_correct * 100.0 / total_bd
    acc_cl = total_cl_correct * 100.0 / total_cl

    info_string = "Clean Acc: {:.4f} total {}| BD Acc: {:.4f} total {}".format(
        acc_cl, total_cl, acc_bd, total_bd
    )

    print(info_string)



def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = ema_param.data * alpha + param.data * (1 - alpha)

def softmax_mse_loss(input_logits, target_logits):

    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') #/ num_classes

def get_current_consistency_weight(epoch):
    return 0.1 * sigmoid_rampup(epoch, 200)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def eval(netC, test_dl, opt, tf_writer, epoch, netG=None, netM=None):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_bd = 0
    total_clean_correct = 0
    total_bd_correct = 0
    if opt.trigger_type == 'sig':
        blend_img = generate(opt)
        #tf_writer.add_image("blend",  blend_img.squeeze(0))

    ac = np.zeros(opt.num_classes)
    asize = np.zeros(opt.num_classes)

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)
            for i in range(len(targets)):
                asize[targets[i].item()] += 1
                if torch.argmax(preds_clean[i]) == targets[i]:
                    ac[targets[i].item()] += 1

            inputs_nt = []
            label_nt = []
            for i in range(len(targets)):
                if targets[i].item() != opt.target_label:
                    inputs_nt.append(inputs[i])
                    label_nt.append(targets[i])
                    
            inputs_nt = torch.stack(inputs_nt, 0)
            total_bd += len(inputs_nt)
            targets_nt = torch.stack(label_nt, 0)

            # Evaluate Backdoor
            with torch.no_grad():
                bs = inputs.shape[0]
                if opt.trigger_type == 'blend':
                    inputs_bd, targets_bd = blend(inputs_nt, targets_nt, opt, tf_writer)
                elif opt.trigger_type == 'patch':
                    inputs_bd, targets_bd = patch(inputs_nt, targets_nt, opt, tf_writer)
                elif opt.trigger_type == 'sig':
                    inputs_bd, targets_bd = sig(inputs_nt, targets_nt, opt)
                elif opt.trigger_type == 'dynamic':
                    inputs_bd, targets_bd = dynamic(inputs_nt, targets_nt, opt, netG, netM)

            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

        #if batch_idx == 1:
            #tf_writer.add_image("Images", inputs[0], global_step=epoch)
            #tf_writer.add_image("Images_bd", inputs_bd[0], global_step=epoch)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_bd = total_bd_correct * 100.0 / total_bd
    #tf_writer.add_scalar("clean", acc_clean, epoch)
    #tf_writer.add_scalar("bd", acc_bd, epoch)

    info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(
            acc_clean, acc_bd
    )
    print(info_string)

    #tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)
    if epoch % 5 == 0:
        for i in range(10):
            print("class {} has a ACC {:.6f}".format(i, (ac[i] / asize[i])))




def mt_aft_train(netC, optimizerC, epoch, train_dl, opt, teacher_model, adv=False, partial=False):
    netC.train()
    criterion_CE = torch.nn.CrossEntropyLoss()
    transforms = PostTensorTransform(opt).to(opt.device)
    '''
    maxiter = 10
    eps = 8. / 255.
    alpha = eps / 8
    '''

    maxiter = opt.maxiter
    eps = opt.eps
    alpha = opt.alpha


    '''
    maxiter = 40
    eps = 2. / 255.
    alpha = 2. / 255.
    '''
    global step

    for batch_idx, (inputs, targets, _) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        total_inputs = transforms(inputs)
        total_targets = targets
        if adv:
            if partial and batch_idx > 30:
                pass
            else:
                netC.eval()
                total_inputs_orig = total_inputs.clone().detach()
                total_inputs.requires_grad = True
                labels = total_targets

                for iteration in range(maxiter):
                    optimx = torch.optim.SGD([total_inputs], lr=1.)
                    optim = torch.optim.SGD(netC.parameters(), lr=1.)
                    optimx.zero_grad()
                    optim.zero_grad()
                    output = netC(total_inputs)
                    pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
                    pgd_loss.backward()

                    total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
                    optimx.step()
                    total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
                    total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
                    # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
                    total_inputs = total_inputs.clone().detach()
                    total_inputs.requires_grad = True

                optimx.zero_grad()
                optim.zero_grad()
                total_inputs.requires_grad = False
                total_inputs = total_inputs.clone().detach()
                netC.train()

        total_preds = netC(total_inputs)
        total_teacher_preds = teacher_model(total_inputs)
        loss_cons = softmax_mse_loss(total_preds, total_teacher_preds)
        loss_ce = criterion_CE(total_preds, total_targets)
        #print(loss_ce)
        #print("-------------------------")


        #while loss_cons > loss_ce * (0.6 ):# (s+1) /50 * 0.8 + 0.2
            #loss_cons = loss_cons * 0.99
        #print(0.1 * loss_cons)
        loss = loss_ce + 0.05 * loss_cons#get_current_consistency_weight(step) * loss_cons
        update_ema_variables(netC, teacher_model, 0.99, step)
        step = step + 1
        loss.backward()
        #print(step)
        optimizerC.step()
    #print(loss)



def rep_saver(netC, train_dl, opt):
    netC.train()

    for batch_idx, (inputs, targets, _) in enumerate(train_dl):

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        #total_inputs = transforms(inputs)
        total_targets = targets
        r1, r2, r3, r4, total_preds = netC.nad_forward(inputs)
        print(r3.shape)
        if batch_idx == 0:
            r1_f = r1.detach().cpu()
            r2_f = r2.detach().cpu()
            r3_f = r3.detach().cpu()
            r4_f = r4.detach().cpu()
        else :
            r1_f = torch.cat((r1_f, r1.detach().cpu()), dim=0)
            r2_f = torch.cat((r2_f, r2.detach().cpu()), dim=0)
            r3_f = torch.cat((r3_f, r3.detach().cpu()), dim=0)
            r4_f = torch.cat((r4_f, r4.detach().cpu()), dim=0)

    r1_f = torch.flatten(r1_f, 1)
    r2_f = r2_f.reshape(r2_f.shape[0], -1)
    r3_f = r3_f.reshape(r3_f.shape[0], -1)
    r4_f = torch.flatten(r4_f, 1)

    return r2_f, r3_f


def eval_warp(
    netC,
    test_dl,
    noise_grid,
    identity_grid,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            inputs_nt = []
            label_nt = []
            for i in range(len(targets)):
                if targets[i].item() != opt.target_label:
                    inputs_nt.append(inputs[i])
                    label_nt.append(targets[i])
            inputs_nt = torch.stack(inputs_nt, 0)
            targets_nt = torch.stack(label_nt, 0)

            inputs_bd = F.grid_sample(inputs_nt, grid_temps.repeat(inputs_nt.shape[0], 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets_nt) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets_nt + 1, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

    acc_clean = total_clean_correct * 100.0 / total_sample
    acc_bd = total_bd_correct * 100.0 / total_sample

    if False:
        inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
        preds_cross = netC(inputs_cross)
        total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

        acc_cross = total_cross_correct * 100.0 / total_sample

        info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross: {:.4f}".format(acc_clean, acc_bd, acc_cross)
    else:
        info_string = "Clean Acc: {:.4f}  Bd Acc: {:.4f} ".format(
            acc_clean, acc_bd
        )

    print(info_string)

def eval_adverarial_attack(netC, test_dl, opt, tf_writer, epoch):
    print(" Eval:")
    netC.eval()

    total_ad_correct = 0
    total_sample = 0
    '''
    maxiter = opt.maxiter
    eps = opt.eps
    alpha = opt.alpha
    '''

    maxiter = 7
    eps = 6. / 255.
    alpha = eps / 5

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]
        total_sample += bs
        total_targets = targets
        total_inputs = inputs
        netC.eval()
        total_inputs_orig = total_inputs.clone().detach()
        total_inputs.requires_grad = True
        labels = total_targets


        for iteration in range(maxiter):
            optimx = torch.optim.SGD([total_inputs], lr=1.)
            optim = torch.optim.SGD(netC.parameters(), lr=1.)
            optimx.zero_grad()
            optim.zero_grad()
            output = netC(total_inputs)
            pgd_loss = -1 * torch.nn.functional.cross_entropy(output, labels)
            pgd_loss.backward()

            total_inputs.grad.data.copy_(alpha * torch.sign(total_inputs.grad))
            optimx.step()
            total_inputs = torch.min(total_inputs, total_inputs_orig + eps)
            total_inputs = torch.max(total_inputs, total_inputs_orig - eps)
            # total_inputs = th.clamp(total_inputs, min=-1.9895, max=2.1309)
            total_inputs = total_inputs.clone().detach()
            total_inputs.requires_grad = True
        

        optimx.zero_grad()
        optim.zero_grad()

        total_inputs.requires_grad = False
        total_inputs = total_inputs.clone().detach()
        preds_ad = netC(total_inputs)

        total_ad_correct += torch.sum(torch.argmax(preds_ad, 1) == total_targets)




    acc_ad = total_ad_correct * 100.0 / total_sample
    #tf_writer.add_scalars("ad", acc_ad, epoch)

    #tf_writer.add_scalars("AD Accuracy", {"AD": acc_ad}, epoch)

    info_string = "Adversarial attack Acc: {:.4f} ".format(
            acc_ad
    )
    print(info_string)


def compute_all_reps(
    model: torch.nn.Sequential,
    data: Union[DataLoader, Dataset],
    *,
    layers: Collection[int],
    flat=False,
) -> Dict[int, np.ndarray]:
    device = get_module_device(model)
    dataloader, dataset = either_dataloader_dataset_to_both(data, eval=True)
    n = len(dataset)
    max_layer = max(layers)
    assert max_layer < len(model)

    reps = {}
    x = dataset[0][0][None, ...].to(device)
    for i, layer in enumerate(model):
        if i > max_layer:
            break
        x = layer(x)
        if i in layers:
            inner_shape = x.shape[1:]
            reps[i] = torch.empty(n, *inner_shape)

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            for i, layer in enumerate(model):
                if i > max_layer:
                    break
                x = layer(x)
                if i in layers:
                    reps[i][start_index : start_index + minibatch_size] = x.cpu()

            start_index += minibatch_size

    if flat:
        for layer in reps:
            layer_reps = reps[layer]
            reps[layer] = layer_reps.reshape(layer_reps.shape[0], -1)

    return reps
