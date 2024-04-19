import copy
import random

import torch
import os
import time
import warnings
import numpy as np
import torchvision
import logging
from torch import nn

from algorithms.server.serverTAPER import TAPER
from algorithms.server.servercifar import servercifar
from algorithms.server.serverdit import DiT_gen_param
from models.MLP import ConvNet_100, ConvNet
from models.model import LocalModel, LocalModel_reparam
from options import args_parser
from util.result_util import set_fixed_seed
from models.ResNet import resnet18, resnet18_reparam

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32

def run(args):

    time_list = []
    model_str = args.model
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # args.model = torchvision.models.resnet18(pretrained=False).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        if args.algorithm == "TAPER" and args.stage == "TAPER-3" :
            args.model = resnet18_reparam(num_classes=args.num_classes).to(args.device)
        elif args.algorithm == "TAPER" and args.stage == "TAPER-2" or args.stage == "TAPER-1":
            args.model = resnet18(num_classes=args.num_classes).to(args.device)
        elif args.algorithm == "server_cifar" and args.stage == "cifar-1":
            args.model = ConvNet_100().to(args.device)
        elif args.algorithm == "server_cifar" and args.stage == "cifar-2":
            args.model = ConvNet().to(args.device)
        # print(args.model)

        # select algorithm
        if args.algorithm == "TAPER":
            args.head = copy.deepcopy(args.model.fc)
            # args.head = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = LocalModel(args.model, args.head)
            print(args.model)
            server = TAPER(args, i)
        elif args.algorithm == "DiT":
            args.head = copy.deepcopy(args.model.fc)
            # args.head = nn.Identity()
            args.model.fc = nn.Identity()
            args.model = LocalModel_reparam(args.model, args.head)
            print(args.model)
            server = DiT_gen_param(args, i)
        elif args.algorithm == "server_cifar":
            print(args.model)
            server = servercifar(args, i)

        else:
            raise NotImplementedError

        if args.stage == "TAPER-3":
            server.train_stage_three()
        elif args.stage == "TAPER-2" or args.stage == "cifar-2":
            server.train_stage_two()
        elif args.stage == "TAPER-1" or args.stage == "cifar-1":
            server.train_stage_one()
        elif args.stage == "DiT-1":
            server.train_dit()
        # server.test_baseline()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    args = args_parser()
    set_fixed_seed(args.seed)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)

    run(args)

