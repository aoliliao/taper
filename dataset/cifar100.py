import abc
import copy
import shutil

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import ujson
from sklearn.model_selection import train_test_split
from torchvision import datasets

# from dataset_util import check


def label_transfer(pre_label):
    pre_label1 = [[] for _ in range(len(pre_label))]
    dic = {19: 11, 29: 15, 0: 4, 11: 14, 1: 1, 86: 5, 90: 18, 28: 3, 23: 10, 31: 11, 39: 5, 96: 17, 82: 2, 17: 9, 71: 10, 8: 18, 97: 8, 80: 16, 74: 16,
           59: 17, 70: 2, 87: 5, 84: 6, 64: 12, 52: 17, 42: 8, 47: 17, 65: 16, 21: 11, 22: 5, 81: 19, 24: 7, 78: 15, 45: 13, 49: 10, 56: 17, 76: 9,
           89: 19, 73: 1, 14: 7, 9: 3, 6: 7, 20: 6, 98: 14, 36: 16, 55: 0, 72: 0, 43: 8, 51: 4, 35: 14, 83: 4, 33: 10, 27: 15, 53: 4, 92: 2, 50: 16,
           15: 11, 18: 7, 46: 14, 75: 12, 38: 11, 66: 12, 77: 13, 69: 19, 95: 0, 99: 13, 93: 15, 4: 0, 61: 3, 94: 6, 68: 9, 34: 12, 32: 1, 88: 8,
           67: 1, 30: 0, 62: 2, 63: 12, 40: 5, 26: 13, 48: 18, 79: 13, 85: 19, 54: 2, 44: 15, 7: 7, 12: 9, 2: 14, 41: 19, 37: 9, 13: 18,
           25: 6, 10: 3, 57: 4, 5: 6, 60: 10, 91: 1, 3: 8, 58: 18, 16: 3}
    for i in range(len(pre_label)):
        pre_label1[i] = dic[pre_label[i]]
    pre_label1 = np.array(pre_label1)
    return pre_label1

def random_groups_cifar100(args):
    CIFAR_PATH = "./dataset/cifar100"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=transform)
    num_workers = 4
    concat_train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    concat_test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    class_list = trainset.classes
    # print(class_list)
    classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train',
               28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle',
               71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose',
               87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree',
               65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake',
               45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark',
               14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal',
               43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
               92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
               66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver',
               61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray',
               30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider',
               85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower',
               37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout',
               3: 'bear', 58: 'pickup_truck', 16: 'can'}

    train_sets = []
    test_sets = []
    task_label = []
    classes_per_group = 10
    client_num = args.num_clients
    seeds = list(range(30))
    # print(seeds)
    for i in range(client_num//10):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        class_indices = torch.randperm(100).tolist()
        groups = []
        for j in range(10):
            group_classes = class_indices[j * classes_per_group: (j + 1) * classes_per_group]
            new_dict = {key: classes[key] for key in group_classes}
            selected_indices = [idx for idx, label in enumerate(trainset.targets) if label in group_classes]
            train_data = torch.utils.data.Subset(trainset, selected_indices)
            train_sets.append(train_data)
            selected_indices = [idx for idx, label in enumerate(testset.targets) if label in group_classes]
            test_data = torch.utils.data.Subset(testset, selected_indices)
            test_sets.append(test_data)
            task_label.append(new_dict)
            groups.append(group_classes)
    print(len(train_sets), len(test_sets))
    print(len(train_sets[0]), len(test_sets[0]))
    return train_sets, test_sets, task_label, concat_train_dataloader, concat_test_dataloader,

def superclass_groups_cifar100(args):
    CIFAR_PATH = "./dataset/cifar100"
    # CIFAR_PATH = "./cifar100"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=transform)
    num_workers = 4
    concat_train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    concat_test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    class_list = trainset.classes
    # print(class_list)
    classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train',
               28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle',
               71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose',
               87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree',
               65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake',
               45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark',
               14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal',
               43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
               92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
               66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver',
               61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray',
               30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider',
               85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower',
               37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout',
               3: 'bear', 58: 'pickup_truck', 16: 'can'}
    mapper = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83],
              [22, 39, 40, 86, 87], [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76],
              [23, 33, 49, 60, 71], [15, 19, 21, 31, 38], [34, 63, 64, 66, 75], [26, 45, 77, 79, 99],
              [2, 11, 35, 46, 98],
              [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [8, 13, 48, 58, 90],
              [41, 69, 81, 85, 89]]
    train_sets = []
    test_sets = []
    task_label = []
    classes_per_group = 2
    client_num = args.num_clients
    seeds = list(range(30))
    # print(seeds)
    for i in range(client_num//10):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        superclass_indices = torch.randperm(20).tolist()
        groups = []
        for j in range(10):
            group_superclasses = superclass_indices[j * classes_per_group: (j + 1) * classes_per_group]
            group_classes = [element for sublist in [mapper[i] for i in group_superclasses] for element in sublist]
            # print(group_classes)
            new_dict = {key: classes[key] for key in group_classes}
            selected_indices = [idx for idx, label in enumerate(trainset.targets) if label in group_classes]
            train_data = torch.utils.data.Subset(trainset, selected_indices)
            train_sets.append(train_data)
            selected_indices = [idx for idx, label in enumerate(testset.targets) if label in group_classes]
            test_data = torch.utils.data.Subset(testset, selected_indices)
            test_sets.append(test_data)
            task_label.append(new_dict)
            groups.append(group_classes)
            # print(new_dict)
    print(len(train_sets), len(test_sets))
    print(len(train_sets[0]), len(test_sets[0]))
    return train_sets, test_sets, task_label, concat_train_dataloader, concat_test_dataloader,

def superclass_diff_groups_cifar100(args):
    CIFAR_PATH = "./dataset/cifar100"
    # CIFAR_PATH = "./cifar100"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform)
    testset = datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=transform)
    num_workers = 4
    concat_train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    concat_test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    classes = {19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train',
               28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle',
               71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose',
               87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree',
               65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake',
               45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark',
               14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal',
               43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange',
               92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo',
               66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver',
               61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray',
               30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider',
               85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower',
               37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout',
               3: 'bear', 58: 'pickup_truck', 16: 'can'}
    mapper = [[4, 30, 55, 72, 95], [1, 32, 67, 73, 91], [54, 62, 70, 82, 92], [9, 10, 16, 28, 61], [0, 51, 53, 57, 83],
              [22, 39, 40, 86, 87], [5, 20, 25, 84, 94], [6, 7, 14, 18, 24], [3, 42, 43, 88, 97], [12, 17, 37, 68, 76],
              [23, 33, 49, 60, 71], [15, 19, 21, 31, 38], [34, 63, 64, 66, 75], [26, 45, 77, 79, 99],
              [2, 11, 35, 46, 98],
              [27, 29, 44, 78, 93], [36, 50, 65, 74, 80], [47, 52, 56, 59, 96], [8, 13, 48, 58, 90],
              [41, 69, 81, 85, 89]]
    train_sets = []
    test_sets = []
    task_label = []
    classes_per_group = 10
    client_num = args.num_clients
    seeds = list(range(100))
    # print(seeds)
    for i in range(client_num//10):
        seed = seeds[i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        superclass_indices = torch.randperm(20).tolist()
        remaining_classes = list(range(20))
        groups = []
        mapper_copy = copy.deepcopy(mapper)
        for sublist in mapper_copy:
            random.shuffle(sublist)
        # print(mapper_copy)
        temp = 0
        for j in range(10):
            if j == 5:
                temp += 1
            group_superclasses = superclass_indices[temp * classes_per_group: (temp + 1) * classes_per_group]
            group_classes = [sublist[0] for sublist in [mapper_copy[i] for i in group_superclasses]]
            # print(group_classes)
            for i in group_superclasses:
                mapper_copy[i].remove(mapper_copy[i][0])
            new_dict = {key: classes[key] for key in group_classes}
            selected_indices = [idx for idx, label in enumerate(trainset.targets) if label in group_classes]
            train_data = torch.utils.data.Subset(trainset, selected_indices)
            train_sets.append(train_data)
            selected_indices = [idx for idx, label in enumerate(testset.targets) if label in group_classes]
            test_data = torch.utils.data.Subset(testset, selected_indices)
            test_sets.append(test_data)
            task_label.append(new_dict)
            groups.append(group_classes)
            # print(new_dict)

    print(len(train_sets), len(test_sets))
    print(len(train_sets[0]), len(test_sets[0]))
    return train_sets, test_sets, task_label, concat_train_dataloader, concat_test_dataloader,

def make_datasets():
    source_folder = "../save/cifar"
    destination_folder = "../save/cifar/train"
    files = [file for file in os.listdir(source_folder) if file.endswith("_1.pth")]

    # 计算要选取的文件数量，这里选取四分之一
    num_files_to_select = len(files) // 4

    # 随机选择四分之一的文件
    # selected_files = random.sample(files, num_files_to_select)

    # 将选取的文件移动到目标文件夹中
    for file in files:
        source_file_path = os.path.join(source_folder, file)
        destination_file_path = os.path.join(destination_folder, file)
        shutil.move(source_file_path, destination_file_path)

if __name__ == '__main__':
    make_datasets()


