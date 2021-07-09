#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
import cv2
import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import traceback


def get_train(train_targets, train, name, save_name):
    input_history_train = np.empty((train_targets.shape[0]), dtype=int)

    train_frames = np.memmap(f"runs/{name}/train_frames_{save_name}", dtype=float, mode='w+', shape=(train_targets.shape[0], 3, 288, 256))
    # train_frames = np.empty((train_targets.shape[0], 3, 288, 256))
    count = 0
    for file in tqdm(train, desc="Building train dataset"):
        data = np.load(file)
        count_old = count
        count += data["input_history"].shape[0]
        input_history_train[count_old:count] = data["input_history"].tolist()

        aux_frame = np.copy(data["car_frames"])
        aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        aux_frame = (2 * (aux_frame / 255)) - 1
        train_frames[count_old:count] = aux_frame
        train_frames.flush()

    return train_frames, input_history_train


def get_test(test_targets, test, name, save_name):
    input_history = np.empty((test_targets.shape[0]), dtype=int)
    test_frames = np.memmap(f"runs/{name}/test_frames_{save_name}", dtype=float, mode='w+', shape=(test_targets.shape[0], 3, 288, 256))
    count = 0
    for file in tqdm(test, desc="Building test dataset"):
        data = np.load(file)
        count_old = count
        count += data["input_history"].shape[0]
        input_history[count_old:count] = data["input_history"].tolist()

        aux_frame = np.copy(data["car_frames"])
        aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        aux_frame = (2 * (aux_frame / 255)) - 1
        test_frames[count_old:count] = aux_frame
        test_frames.flush()

    return test_frames, input_history


def test_list(test):
    test_targets = []
    for file in test:
        data = np.load(file)
        test_targets += data["input_history"].tolist()
    test_targets = np.array(test_targets).flatten()
    print("test_targets shape", test_targets.shape)
    return test_targets


def train_list(train):
    train_targets = []
    for file in train:
        data = np.load(file)
        train_targets += data["input_history"].tolist()
    train_targets = np.array(train_targets).flatten()

    print("train_targets shape", train_targets.shape)

    return train_targets


def rew_list(train, name):
    rew = []
    for file in train:
        data = np.load(file)
        aux = data["r_history"]
        if aux[-1] < 900:
            print(file)
            exit()
        rew += [aux[-1]]

    fig = plt.figure(figsize=(5, 5))

    plt.title("Total reward of training dataet")
    plt.boxplot(rew)

    plt.savefig(f"Plots/total_reward_{name}.png", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    plt.show()


def get_equalized(input_history_train, train_frames, name, save_name):
    order, indices = np.unique(input_history_train, return_counts=True)
    # print("Values", indices, order)
    zipped_lists = zip(indices, order)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    indices, order = [list(tuple) for tuple in tuples]
    # print("Values", indices[::-1], order[::-1])  # Values [4703 1642 1036  559] [4 3 2 1]
    top = int(indices[::-1][2] * 1.25)
    accus = [i if i < top else top for i in indices]
    accu = np.sum(accus)
    # print("accumulated", accus, order)  # accumulated [559, 1036, 1139, 1139] [1 2 3 4]
    # print("accumulated sum", accu)

    equalized_commands = np.append(input_history_train[np.where(input_history_train == 1)][:top], input_history_train[np.where(input_history_train == 2)][:top])

    equalized_commands = np.append(equalized_commands, input_history_train[np.where(input_history_train == 3)][:top])
    equalized_commands = np.append(equalized_commands, input_history_train[np.where(input_history_train == 4)][:top])

    # print(train_frames[np.where(input_history_train == 1)][:int(indices[::-1][2] * 1.1)].shape)
    # print(train_frames[np.where(input_history_train == 2)][:int(indices[::-1][2] * 1.1)].shape)
    # print(train_frames[np.where(input_history_train == 3)][:int(indices[::-1][2] * 1.1)].shape)
    # print(train_frames[np.where(input_history_train == 4)][:int(indices[::-1][2] * 1.1)].shape)
    #
    # equalized_frames = np.vstack((train_frames[np.where(input_history_train == 1)][:int(indices[::-1][2] * 1.1)], train_frames[np.where(input_history_train == 2)][:int(indices[::-1][2] * 1.1)]))
    #
    # equalized_frames = np.vstack((equalized_frames, train_frames[np.where(input_history_train == 3)][:int(indices[::-1][2] * 1.1)]))
    # equalized_frames = np.vstack((equalized_frames, train_frames[np.where(input_history_train == 4)]))

    zipped_lists = zip(order, accus)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    order, accus = [list(tuple) for tuple in tuples]
    # print("accumulated", accus, order)  # accumulated [559, 1036, 1139, 1139] [1 2 3 4]
    sumed = np.cumsum(accus)
    equalized_frames = np.memmap(f"runs/{name}/equalized_frames_{save_name}", dtype=float, mode='w+', shape=(accu, 3, 288, 256))

    equalized_frames[0:sumed[0]] = train_frames[np.where(input_history_train == 1)][:top]
    equalized_frames.flush()
    equalized_frames[sumed[0]:sumed[1]] = train_frames[np.where(input_history_train == 2)][:top]
    equalized_frames.flush()

    equalized_frames[sumed[1]:sumed[2]] = train_frames[np.where(input_history_train == 3)][:top]
    equalized_frames.flush()

    equalized_frames[sumed[2]:] = train_frames[np.where(input_history_train == 4)][:top]
    equalized_frames.flush()

    return equalized_commands, equalized_frames


def main():
    name = 'New_runs_with_car_corrected'
    # save_name = "Small_50"
    # file_list = np.array(glob.glob(f"runs/{name}/**/discrete_histories.npz"))[:50]

    try:
        save_name = "Small_50" #"Complete"
        file_list = np.array(glob.glob(f"runs/{name}/**/discrete_histories.npz"))[:50]

        print("File list shape", file_list.shape)

        np.random.shuffle(file_list)
        train = file_list[:int(file_list.shape[0] * .8)]
        test = file_list[int(file_list.shape[0] * .8):]

        print("train ", train.shape)
        print("test ", test.shape)

        np.savez(f"runs/{name}/Dataset_div", train_files=train, test_files=test)

        train_targets = train_list(train)
        rew_list(train, name)
        test_targets = test_list(test)

        fig = plt.figure(figsize=(15, 5))
        X = np.arange(4)

        plt.title(f"Dataset target distribution for: \nTraining dataset {train_targets.shape[0]} examples, from {train.shape[0]} runs \nTest dataset {test_targets.shape[0]} examples, from {test.shape[0]} runs ")
        plt.bar(X - 0.125, np.unique(test_targets, return_counts=True)[1], color='b', width=0.25)
        plt.bar(X + 0.125, np.unique(train_targets, return_counts=True)[1], color='g', width=0.25)

        directions = ["Left", 'Right', 'Accelerate', "Brake"]
        plt.legend(labels=['Test', 'Train'])
        plt.xticks(X, directions)
        plt.savefig(f"Plots/dataset_statistics_{name}.png", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        plt.show()

        # input_history_train = []
        #
        # data = np.load(train[0])
        # # input_history_train.append(data["input_history"]) #
        # input_history_train += data["input_history"].tolist()
        #
        # aux_frame = np.copy(data["car_frames"])
        #
        # aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        # train_frames = (2 * (aux_frame / 255)) - 1
        #
        # for file in tqdm(train[1:], desc="Building train dataset"):
        #     data = np.load(file)
        #     # input_history_train.append(data["input_history"]) #
        #     input_history_train += data["input_history"].tolist()
        #     aux_frame = np.copy(data["car_frames"])
        #
        #     aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        #     aux_frame = (2 * (aux_frame / 255)) - 1
        #     train_frames = np.vstack((train_frames, aux_frame))
        #
        # input_history_train = np.array(input_history_train).flatten()

        train_frames, input_history_train = get_train(train_targets, train, name, save_name)

        # input_history = []
        #
        # data = np.load(test[0])
        # # input_history.append(data["input_history"]) #
        # input_history += data["input_history"].tolist()
        #
        # aux_frame = np.copy(data["car_frames"])
        #
        # aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        # test_frames = (2 * (aux_frame / 255)) - 1
        #
        # for file in tqdm(test[1:], desc="Building test dataset"):
        #     data = np.load(file)
        #     # input_history.append(data["input_history"]) #
        #     input_history += data["input_history"].tolist()
        #
        #     aux_frame = np.copy(data["car_frames"])
        #
        #     aux_frame = np.transpose(aux_frame, axes=[0, 3, 1, 2])
        #     aux_frame = (2 * (aux_frame / 255)) - 1
        #     test_frames = np.vstack((test_frames, aux_frame))
        #
        # input_history = np.array(input_history).flatten()

        test_frames, input_history = get_test(test_targets, test, name, save_name)

        print("input_history_train shape", input_history_train.shape)
        print("train_frames shape", train_frames.shape, "min", train_frames.min(), "max", train_frames.max())

        print("input_history_test shape", input_history.shape)
        print("test_frames shape", test_frames.shape, "min", test_frames.min(), "max", test_frames.max())

        # print("Saving file ...")
        # np.savez_compressed(f"runs/{name}/Dataset_complete_{save_name}", train_files=train, test_files=test, input_history_train=input_history_train, train_frames=train_frames, input_history_test=input_history, test_frames=test_frames)
        # print(f"runs/{name}/Dataset_complete_{save_name} saved")

        print("Balancing dataset...")
        equalized_commands, equalized_frames = get_equalized(input_history_train, train_frames, name, save_name)
        fig = plt.figure(figsize=(15, 5))
        X = np.arange(4)

        plt.title(f"Dataset target distribution for: Training dataset {train_targets.shape[0]} examples, from {train.shape[0]} runs \nBalanced dataset {equalized_commands.shape[0]} examples")
        plt.bar(X - 0.25, np.unique(input_history_train, return_counts=True)[1], color='g', width=0.25)
        plt.bar(X + 0.00, np.unique(equalized_commands, return_counts=True)[1], color='r', width=0.25)
        plt.bar(X + 0.25, np.unique(test_targets, return_counts=True)[1], color='b', width=0.25)



        directions = ["Left", 'Right', 'Accelerate', "Brake"]
        plt.legend(labels=['Train', 'Balanced train', 'Test'])
        plt.xticks(X, directions)
        plt.savefig(f"Plots/dataset_statistics_balanced_{name}.png", dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
        plt.show()

        print("equalized_commands shape", equalized_commands.shape)

        print("equalized_frames shape", equalized_frames.shape, "min", equalized_frames.min(), "max", equalized_frames.max())

        print("Saving balanced dataset ...")
        np.savez_compressed(f"runs/{name}/Dataset_equalized_{save_name}", input_history_train=equalized_commands, train_frames=equalized_frames, input_history_test=input_history, test_frames=test_frames)

        if os.path.exists(f"runs/{name}/train_frames_{save_name}"):
            os.remove(f"runs/{name}/train_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists(f"runs/{name}/test_frames_{save_name}"):
            os.remove(f"runs/{name}/test_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists(f"runs/{name}/equalized_frames_{save_name}"):
            os.remove(f"runs/{name}/equalized_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

    except:

        if os.path.exists(f"runs/{name}/train_frames_{save_name}"):
            os.remove(f"runs/{name}/train_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists(f"runs/{name}/test_frames_{save_name}"):
            os.remove(f"runs/{name}/test_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists(f"runs/{name}/equalized_frames_{save_name}"):
            os.remove(f"runs/{name}/equalized_frames_{save_name}")
            print("File deleted")
        else:
            print("The file does not exist")

        print(traceback.print_exc())
        sys.exit(0)


if __name__ == "__main__":
    main()
