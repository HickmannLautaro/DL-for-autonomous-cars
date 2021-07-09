#!/usr/bin/env python

import numpy as np
from pyglet.window import Window
from simulation.gym import gym
from simulation.render_utils import show_stuff
import time
import torch
import sys
sys.path.insert(0,'..')
from predict import prepare, generate
from enum import Enum



def main(log=True, classifier=None, device=None, car=None, dim=256, style_transfer=False, a_cl=1, b_cl=0):

    # if car is None:
    #     car = np.load("car_positions.npz")
    print("Simulation started")
    env = gym.make('CarRacing-v0')
    # dual_color_hist_short
    if style_transfer:
        model = prepare(name="dual_color_hist_tree", mode="FastCUT", path = "../styletransfer/cut/checkpoints")  # continued_road_FastCUT
    # dual_color_hist_short
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if classifier is None:
        print("Using default")
        classifier = torch.load("out/old_dataset/LeNet_mod_bz_128_lr_0.0001_ep_30/model_ep_3.pth").to(device)
    elif type(classifier) == str:
        print(f"Loading {classifier}")
        classifier = torch.load(classifier).to(device)



    classifier.eval()

    r_history = []  # np.array([-100])
    frames = []  # env.reset()[None,:,:,:]
    road_frames = []  # env.reset()[None,:,:,:]
    env.reset()

    width = 1000
    height = 800
    win = Window(width=width, height=height)

    a = np.array([0.0, 0.0, 0.0])  # isOpen, restert, Left, Right, accelerate, stop
    input_history = []  # np.array([0.0, 0.0, 0.0])

    # env.viewer.window.on_key_press = key_press
    # env.viewer.window.on_key_release = key_release

    repetition = 0
    path_list = []

    env.reset()
    r_history = []  # np.array([-100])
    frames = []  # env.reset()[None,:,:,:]
    seg_frames = []

    road_frames = []  # env.reset()[None,:,:,:]
    input_history = []
    total_reward = 0.0
    steps = 0
    break_count = 0
    left_count = 0
    right_count = 0
    accel_count = 0
    command = np.array([3])
    while True:

        start_time = time.time()

        a = np.array([0.0, 0.0, 0.0])
        if accel_count == 0:
            if steps ==0:
                a = np.array([0.0, 1.0, 0.0])
                start = False
            else:
                if command == 0:
                    a[0] = -1.0  # -1.0
                    break_count = 0
                    left_count +=1
                    right_count = 0
                elif command == 1:
                    a[0] = 1.0  # +1.0
                    break_count = 0
                    left_count = 0
                    right_count += 1
                elif command == 2:
                    a[1] = +1.0  # +0.9
                    break_count = 0
                    left_count = 0
                    right_count = 0
                elif command == 3:
                    break_count += 1
                    left_count = 0
                    right_count = 0
                    a[2] = +0.8
                else:
                    print("Wrong input")
            if break_count >= 5 or left_count >= 25 or right_count >= 25:
                break_count = 0
                left_count = 0
                right_count = 0

                a = np.array([0.0, 1.0, 0.0])
                accel_count = 10
                command = np.array([2])
        else:
            accel_count -= 1
            command = np.array([2])
            a = np.array([0.0, 1.0, 0.0])

        if steps > 0:
            input_history.append(command) # Delayed by one because of the accel and so controls
        frame, r, done, info, car_frame, broke = env.step(a)

        new_frame = np.copy(frame)

        # styletransfer
        if style_transfer:
            status_bar = new_frame[257:]
            new_frame = new_frame[:256, :256]
            new_frame = generate(model, new_frame, a=a_cl, b=b_cl)
            new_frame = np.vstack((new_frame, status_bar))


        # Add car
        new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [204, 0, 0], axis=2) == 3)] = [204, 0, 0]
        new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [76, 76, 76], axis=2) == 3)] = [76, 76, 76]
        new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [0, 0, 0], axis=2) == 3)] = [0, 0, 0]

        # Classifer
        aux_frame = np.copy(new_frame)
        aux_frame = np.transpose(aux_frame[:dim, :, :], axes=[2, 0, 1])
        aux_frame = (2 * (aux_frame / 255)) - 1
        aux_frame = aux_frame[np.newaxis, :]
        x = torch.from_numpy(aux_frame)
        x = x.to(device=device, dtype=torch.float)

        if accel_count ==0:
            pred = classifier(x)
            command = pred.argmax(axis=1).cpu().numpy()


        total_reward += r
        r_history.append(total_reward)  # = np.append(r_history,r)

        if log:
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1

        fps = 1.0 / (time.time() - start_time)

        if log:
            while fps > 61:
                fps = 1.0 / (time.time() - start_time)

        if steps == 1:
            cmd = None
        else:
            cmd = input_history

        show_stuff(win, np.copy(new_frame), width, height, total_reward, fps, car, True, steps, cmd, discrete=True)

        # if done or restart or isopen == False:
        if done or broke:
            break
    r_history = np.array(r_history)
    input_history = np.array(input_history)

    # road_frames = np.array(road_frames)

    env.close()
    win.close()
    return r_history[:-1], input_history


if __name__ == "__main__":
    main(True,style_transfer=True, a_cl=1, b_cl=0, training_mode=False)
