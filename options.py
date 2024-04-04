# -*-coding:utf-8-*-
import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)

    parser.add_argument('-dev', "--device", type=str, default="cuda", )
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="domainnet")
    parser.add_argument('-nb', "--num_classes", type=int, default=345)

    parser.add_argument('-m', "--model", type=str, default="resnet")
    # parser.add_argument('-p', "--head", type=str, default="cnn")

    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=125)
    parser.add_argument('-ls', "--local_steps", type=int, default=20)
    parser.add_argument('-algo', "--algorithm", type=str, default="TAPER")
    parser.add_argument('-stage', "--stage", type=str, default="TAPER-1")
    parser.add_argument('-per_node', "--per_node", type=int, default=5,
                        help="number of node for each domain")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('-mom', "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument('-wed', "--weight_decay", type=float, default=0.0001, help="weight_decay")


    # not used
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=4,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('-com', "--comment", type=str, default="")
    parser.add_argument('--alpha', type=float, default=1.0, help='percentage of dataset to train')

    args = parser.parse_args()
    return args
