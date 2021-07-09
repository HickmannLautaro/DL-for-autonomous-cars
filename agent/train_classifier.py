# USAGE

import os
import sys
import time
import traceback
from zipfile import BadZipfile

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from visdom import Visdom

import AD_test
# import the necessary packages
from classifier_models.classifier_models import Nvidia_model, LeNet, LeNet_mod, Nvidia_model_resized, LeNet_resized, LeNet_resized_cons, Nvidia_model_resized_cons, Nvidia_model_resized_cons_no_bn, LeNet_resized_cons_no_bn

import create_st_data
def get_train_data(BATCH_SIZE, path, dim):
    print("[INFO] Start training phase...")
    # define the train and val splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 1 - TRAIN_SPLIT

    # load the KMNIST dataset
    print("[INFO] loading the dataset...")

    data = np.load(f"../runs/{path}.npz", mmap_mode='r')
    train_labels = data['input_history_train'] - 1
    train_frames = data['train_frames'][:, :, :dim, :]
    if not os.path.exists("out/tmp/"):
        os.makedirs("out/tmp/")

    train_frames_mem = np.memmap("out/tmp/train_data", dtype=float, mode='w+', shape=train_frames.shape)
    train_frames_mem[:, :, :] = train_frames[:, :, :]
    train_frames_mem.flush()

    print(train_frames_mem.shape)

    print("[INFO] converting to pytorch dataset...")

    trainData = TensorDataset(torch.from_numpy(train_frames_mem), torch.from_numpy(train_labels))

    # calculate the train/validation split
    # print("[INFO] generating the train/validation split...")
    # numTrainSamples = int(np.around(len(trainData) * TRAIN_SPLIT))
    # numValSamples = int(np.around(len(trainData) * VAL_SPLIT))
    # (trainData, valData) = random_split(trainData,
    #                                     [numTrainSamples, numValSamples],
    #                                     generator=torch.Generator())

    # initialize the train, validation, and test data loaders
    trainDataLoader = DataLoader(trainData, shuffle=True,
                                 batch_size=BATCH_SIZE)
    # valDataLoader = DataLoader(valData, shuffle=True, batch_size=BATCH_SIZE)

    # calculate the train/validation split

    return trainDataLoader  # , testDataLoader, test_labels


def get_test_data(path, dim, BATCH_SIZE):
    print("[INFO] loading the dataset...")
    data = np.load(f"../runs/{path}.npz", mmap_mode='r')

    test_labels = data['input_history_test'] - 1
    test_frames = data['test_frames'][:, :dim, :]

    print("[INFO] converting to pytorch dataset...")

    test_frames_mem = np.memmap("out/tmp/test_data", dtype=float, mode='w+', shape=test_frames.shape)
    test_frames_mem[:, :, :] = test_frames[:, :, :]
    test_frames_mem.flush()

    testData = TensorDataset(torch.from_numpy(test_frames_mem), torch.from_numpy(test_labels))

    testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)
    return testDataLoader, test_labels


def train(INIT_LR, BATCH_SIZE, EPOCHS, device, path_load_data, path_save_model, dim, log_inter, mod_func, style_transfer, a_cl, b_cl):
    car = np.load("../car_positions.npz")

    trainDataLoader = get_train_data(BATCH_SIZE, path_load_data, dim)  # valDataLoader, valTargets

    print("[INFO] Loading validatio ndataset...")
    valDataLoader, valTargets = get_test_data(path_load_data, dim, BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    # initialize the LeNet model
    print("[INFO] initializing the LeNet model...")
    model = mod_func(
        numChannels=3,
        classes=4, dim=dim).to(device)

    print(model)
    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)
    lossFn = nn.NLLLoss()

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        'report': []
    }

    viz = Visdom()

    if not viz.check_connection(): sys.exit("Start Visdom:\n python -m visdom.server")

    # measure how long training is going to take
    print("[INFO] training the network...")
    startTime = time.time()
    r_histos = []
    # loop over our epochs
    for e in range(0, EPOCHS):
        epoch_start_time = time.time()
        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0

        # initialize the number of correct predictions in the training
        # and validation step
        trainCorrect = 0
        valCorrect = 0

        # loop over the training set
        for (x, y) in trainDataLoader:
            # send the input to the device

            (x, y) = (x.to(device=device, dtype=torch.float), y.to(device=device))

            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)

            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            preds = []

            # loop over the validation set
            for (x, y) in valDataLoader:
                # send the input to the device
                (x, y) = (x.to(device=device, dtype=torch.float), y.to(device))

                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossFn(pred, y)

                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

                preds.extend(pred.argmax(axis=1).cpu().numpy())

                # generate a classification report

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)

        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        report = classification_report(valTargets, np.array(preds), target_names=["Left", 'Right', 'Accelerate', "Brake"], output_dict=True)
        H['report'].append(report.copy())

        viz.line([[trainCorrect, valCorrect, report['accuracy'], report['Left']['f1-score'], report['Right']['f1-score'], report['Accelerate']['f1-score'], report['Brake']['f1-score']]], [e], win=f"accuracy_{path_save_model}",
                 opts=dict(title=f"train&val accuracy {path_save_model}", legend=['train_acc', 'val_acc', 'rep_acc', 'Left_f1-score', 'Right_f1-score', 'Accelerate_f1-score', 'Brake_f1-score']), update='append')

        viz.line([[avgTrainLoss.cpu().detach().numpy(), avgValLoss.cpu().detach().numpy()]], [e], win=f"loss_{path_save_model}", opts=dict(title=f"train&val loss {path_save_model}", legend=['train_loss', 'val_loss']), update='append')

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}, Time {:.3f} sec ,Train loss: {:.6f}, Train accuracy: {:.4f} Val loss: {:.6f}, Val accuracy: {:.4f}".format(e, EPOCHS, time.time() - epoch_start_time, avgTrainLoss, trainCorrect, avgValLoss, valCorrect))
        if e % log_inter == 0:
            sim_start_time = time.time()
            print("[Simulation] Simulation 1")
            r_history, input_history = AD_test.main(False, model, device, car, dim, style_transfer, a_cl[0], b_cl[0])
            print("[Simulation] Simulation 2")
            r_history_2, input_history_2 = AD_test.main(False, model, device, car, dim, style_transfer, a_cl[1], b_cl[1])

            viz.line(X=np.arange(r_history.shape[0]), Y=r_history, win=f"rewards_{path_save_model}", update='append', name=f"epoch {e} 1", opts=dict(title=f"Reward per epoch {path_save_model}"))
            viz.line(X=np.arange(r_history_2.shape[0]), Y=r_history_2, win=f"rewards_{path_save_model}", update='append', name=f"epoch {e} 2", opts=dict(title=f"Reward per epoch {path_save_model}"))

            # np.mean([r_history.max() if r_history.max() > 850 else r_history[-1], r_history_2.max() if r_history_2.max() > 850 else r_history_2[-1]])
            viz.line([np.mean([r_history[-1], r_history_2[-1]])], [e], win=f"rewards_cum_{path_save_model}", update='append', opts=dict(title=f"Total reward over epochs {path_save_model}", legend=['total reward']))

            r_histos.append(r_history)
            r_histos.append(r_history_2)
            if not os.path.exists(f"./out/{path_save_model}/"):
                os.makedirs(f"./out/{path_save_model}/")

            torch.save(model, f"./out/{path_save_model}/model_ep_{e}.pth")
            print(f"[Simulation] Simulations done, Time {int(time.time() - sim_start_time)}, Model saved (./out/{path_save_model}/model_ep_{e}.pth)")

    print("[Simulation] Simulation 1")
    r_history, input_history = AD_test.main(False, model, device, car, dim, style_transfer, a_cl[0], b_cl[0])
    print("[Simulation] Simulation 2")

    r_history_2, input_history_2 = AD_test.main(False, model, device, car, dim, style_transfer, a_cl[1], b_cl[1])

    viz.line(X=np.arange(r_history.shape[0]), Y=r_history, win=f"rewards_{path_save_model}", update='append', name=f"epoch {e} 1", opts=dict(title=f"Reward per epoch {path_save_model}"))
    viz.line(X=np.arange(r_history_2.shape[0]), Y=r_history_2, win=f"rewards_{path_save_model}", update='append', name=f"epoch {e} 2", opts=dict(title=f"Reward per epoch {path_save_model}"))

    viz.line([np.mean([r_history[-1], r_history_2[-1]])], [e], win=f"rewards_cum_{path_save_model}", update='append', opts=dict(title=f"Total reward over epochs {path_save_model}", legend=['total reward']))
    r_histos.append(r_history)
    r_histos.append(r_history_2)
    # finish measuring how long training took
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    if not os.path.exists(f"./out/{path_save_model}/"):
        os.makedirs(f"./out/{path_save_model}/")
    torch.save(model, f"./out/{path_save_model}/model_ep_{e}.pth")
    print(f"Model saved (./out/{path_save_model}/model_ep_{e}.pth)")

    np.savez(f"./out/{path_save_model}/statistics.npz", r_histos=np.array(r_histos), history=H)
    # torch.onnx.export(model, x, f"./out/{path_save_model}/onnx_model.onnx", input_names=['input'], output_names=['output'], export_params=True, do_constant_folding=True, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    if os.path.exists("out/tmp/test_data"):
        os.remove("out/tmp/test_data")
        print("File deleted")
    else:
        print("The file does not exist")

    if os.path.exists("out/tmp/train_data"):
        os.remove("out/tmp/train_data")
        print("File deleted")
    else:
        print("The file does not exist")
    return model


def evaluate(model, BATCH_SIZE, device, path, path_save_model, dim):
    # we can now evaluate the network on the test set

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("out/new_dataset/style_transfer/Nvidia_model_resized_cons_no_bn_bz_128_lr_0.0001_ep_30/style_brown/model_ep_24.pth").to(device)
    print("[INFO] evaluating network...")

    testDataLoader, test_labels = get_test_data(path, dim, BATCH_SIZE)

    # calculate the train/validation split
    print("[INFO] generating the train/validation split...")

    # initialize the train, validation, and test data loaders

    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []

        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device=device, dtype=torch.float)

            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())

    # generate a classification report
    print(classification_report(test_labels,
                                np.array(preds), target_names=["Left", 'Right', 'Accelerate', "Brake"]))

    # serialize the model to disk

    for (x, y) in testDataLoader:
        x = x.to(device=device, dtype=torch.float)

        torch.onnx.export(model, x, f"./out/{path_save_model}/onnx_model.onnx", input_names=['input'], output_names=['output'], export_params=True, do_constant_folding=True, dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}, opset_version=11)
        break

    if os.path.exists("out/tmp/test_data"):
        os.remove("out/tmp/test_data")
        print("File deleted")
    else:
        print("The file does not exist")


def train_classifier(path_save_model, path_load_data, INIT_LR, BATCH_SIZE, EPOCHS, dim, log_inter, mod_func, style_transfer=False, a_cl=[1, 1], b_cl=[0, 0]):
    print("#" * 500)
    print(f"Starting {path_save_model} with {path_load_data}")
    # set the device we will be using to train the model
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train(INIT_LR, BATCH_SIZE, EPOCHS, device, path_load_data, path_save_model, dim, log_inter, mod_func, style_transfer, a_cl, b_cl)
    # model = torch.load(path_save_model).to(device)

    # evaluate(model, BATCH_SIZE, device, path_load_data, path_save_model, dim)
    print(f"Ended {path_save_model} with {path_load_data}")
    print("#" * 500)


def Hyperparameter_search(EPOCHS):
    # define training hyperparameters
    log_inter = 3
    path_load_data = "New_runs_with_car_corrected/Dataset_equalized_Small_50"  # Complete"

    for BATCH_SIZE in [128, 256, 64]:
        print("*" * 500)
        print(f"BATCH SIZE {BATCH_SIZE}")
        for INIT_LR in [ 1e-4, 1e-3]:

            print("-" * 500)
            print(f"LEARNING RATE {INIT_LR}")

            mod_funcs = [Nvidia_model, LeNet, LeNet_mod, Nvidia_model_resized, LeNet_resized, LeNet_resized_cons, Nvidia_model_resized_cons, Nvidia_model_resized_cons_no_bn, LeNet_resized_cons_no_bn,
                         Nvidia_model, LeNet, LeNet_mod, Nvidia_model_resized, LeNet_resized, LeNet_resized_cons, Nvidia_model_resized_cons, Nvidia_model_resized_cons_no_bn, LeNet_resized_cons_no_bn
                         ]
            dims = [256, 256, 256, 256, 256, 256, 256, 256, 256, 288, 288, 288, 288, 288, 288, 288, 288, 288]
            path_save_models = [f"Nvidia_model_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_mod_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_model_resized_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_resized_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_resized_cons_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_model_resized_cons_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_model_resized_cons_no_bn_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_resized_cons_no_bn_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_model_bar_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_bar_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_mod_bar_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_model_resized_bar_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_bar_resized_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_bar_resized_cons_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_bar_model_resized_cons_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"Nvidia_bar_model_resized_cons_no_bn_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}",
                                f"LeNet_bar_resized_cons_no_bn_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}"]
            #    out/new_dataset/Nvidia_model_resized_cons_no_bn_bz_128_lr_0.0001_ep_30/model_ep_3.pth 789.926 index 3

            for (mod_func, path_save_model, dim) in zip(mod_funcs, path_save_models, dims):
                path_save_model = "new_dataset/" + path_save_model
                if not os.path.exists(f"./out/{path_save_model}/"):

                    try:
                        train_classifier(path_save_model, path_load_data, INIT_LR, BATCH_SIZE, EPOCHS, dim, log_inter, mod_func)
                    except (RuntimeError, BadZipfile) as e:
                        if 'out of memory' in str(e):
                            print(f"[OOM] skipping run {path_save_model}")
                            if not os.path.exists(f"./out/{path_save_model}/"):
                                os.makedirs(f"./out/{path_save_model}/")
                                os.makedirs(f"./out/{path_save_model}/OOM")

                            continue
                        elif 'BadZipFile' in str(e):
                            print(f'[Eroor] bad zipfile restarting')
                            train_classifier(path_save_model, path_load_data, INIT_LR, BATCH_SIZE, EPOCHS, dim, log_inter, mod_func)



def train_style_transfer(EPOCHS):
    # define training hyperparameters


    INIT_LR = 0.001
    BATCH_SIZE = 128

    log_inter = 3

    mod_func = Nvidia_model

    dim = 256
    for style in ["green", "brown", "mixed"]:

        path_load_data = f"New_runs_with_car_corrected/Dataset_equalized_{style}_style"
        if style == 'green':
            a = [1, 1]
            b = [0, 0]
        elif style == "brown":
            a = [0, 0]
            b = [1, 1]
        else:
            a = [0, 1]
            b = [1, 0]
        path_save_model = f"new_dataset/style_transfer/Nvidia_model_bz_{BATCH_SIZE}_lr_{INIT_LR}_ep_{EPOCHS}/style_{style}"

        if not os.path.exists(f"./out/{path_save_model}/"):
            try:
                train_classifier(path_save_model, path_load_data, INIT_LR, BATCH_SIZE, EPOCHS, dim, log_inter, mod_func, style_transfer=True, a_cl=a, b_cl=b)
            except BadZipfile as e:
                print(f'[Error] bad zipfile restarting')
                train_classifier(path_save_model, path_load_data, INIT_LR, BATCH_SIZE, EPOCHS, dim, log_inter, mod_func, style_transfer=True, a_cl=a, b_cl=b)

        else:
            print(f"Exists already ./out/{path_save_model}/")


if __name__ == "__main__":

    EPOCHS = 30
    try:
        train_style_transfer(EPOCHS)
        #Hyperparameter_search(EPOCHS)


    except:

        if os.path.exists("out/tmp/test_data"):
            os.remove("out/tmp/test_data")
            print("File deleted")
        else:
            print("The file does not exist")

        if os.path.exists("out/tmp/train_data"):
            os.remove("out/tmp/train_data")
            print("File deleted")
        else:
            print("The file does not exist")

        print(traceback.print_exc())
        sys.exit(0)
