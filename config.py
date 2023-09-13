import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--attack_mode", type=str, default="all2one")

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=0.01
                        )
    #0.003 for patch
    #0.001 for warp

    parser.add_argument("--schedulerC_milestones", type=list, default=[30, 60, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--num_workers", type=float, default=6)

    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--pc", type=float, default=0.02)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )  # scale grid values to avoid pixel values going out of [-1, 1]. For example, grid-rescale = 0.98

    parser.add_argument("--remove_one_label", type=bool, default=False)

    parser.add_argument("--trigger_type", type=str, default="blend")
    parser.add_argument("--adversarial_maxiter", type=int, default=15)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--eps", type=float, default=2. / 255.)
    parser.add_argument("--alpha", type=float, default=0.5 / 255.)
    parser.add_argument("--model_path", type=str, default="./pt/cifar10_all2one_morph.pth.tar")

    return parser

