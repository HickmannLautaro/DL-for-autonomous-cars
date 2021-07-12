#!/usr/bin/env python

import time
from datetime import datetime
from enum import Enum

import numpy as np
from pyglet.window import key, Window

import time_limit
from styletransfer.predict import prepare, generate
from render_utils import show_stuff
import car_racing
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class StyleStates(Enum):
    SIMULATED = 1
    STYLE_1 = 2
    STYLE_2 = 3

    SPLIT_1 = 4
    SPLIT_2 = 5
    MIXED = 6

style_state = StyleStates.SIMULATED
st = np.array([1.0, 0, 0])


def main():

    run_name = "Trial_1"

    car = np.load("car_positions.npz")
    env = car_racing.CarRacing()
    env = time_limit.TimeLimit(env, max_episode_steps=1000)

    #dual_color_hist_short
    model = prepare(name="new_trees/styletransfer_MSE_histo_10_edges_10", mode="FastCUT", path = "../styletransfer/cut/checkpoints") # continued_road_FastCUT


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r_history = []  # np.array([-100])
    frames = []  # env.reset()[None,:,:,:]
    road_frames = []  # env.reset()[None,:,:,:]
    env.reset()

    width = 1000
    height = 800
    win = Window(width=width, height=height)

    commands = [True, False, False]
    extra_commands = [False, False, False] # Save, endless, GradCAM
    # isopen, restart
    # isopen = True
    # restart = False

    a = np.array([0.0, 0.0, 0.0])
    input_history = []  # np.array([0.0, 0.0, 0.0])
    # CLASSIFIER_LIST = np.array([[None, "None"],
    #                             ["classifier/out/new_dataset/Nvidia_model_bz_128_lr_0.001_ep_30/model_ep_9.pth", "OpenAI"],
    #                             ["classifier/out/new_dataset/style_transfer/Nvidia_model_resized_cons_no_bn_bz_128_lr_0.0001_ep_30/style_green/model_ep_24.pth", "Brown"],
    #                             ["classifier/out/new_dataset/style_transfer/Nvidia_model_resized_cons_no_bn_bz_128_lr_0.0001_ep_30/style_brown/model_ep_15.pth", "Green"],
    #                             ["classifier/out/new_dataset/style_transfer/Nvidia_model_resized_cons_no_bn_bz_128_lr_0.0001_ep_30/style_mixed/model_ep_18.pth", "Mixed"]])

    CLASSIFIER_LIST = np.array([[None, "None"],
                                ["../agent/out/new_dataset/Nvidia_model_bz_128_lr_0.001_ep_30/model_ep_9.pth", "OpenAI"],
                                ["../agent/out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_brown/model_ep_6.pth", "Brown"],
                                ["../agent/out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_green/model_ep_24.pth", "Green"],
                                ["../agent/out/new_dataset/style_transfer/Nvidia_model_bz_128_lr_0.001_ep_30/style_mixed/model_ep_0.pth", "Mixed"]])


    roll_ind = np.array([0, 0])
    def key_press(k, mod):
        # global restart
        # global isopen
        global style_state
        global step
        if k == 0xFF0D:
            restart = True
            commands[1] = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +0.9
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
        if k == key.ESCAPE:
            commands[0] = False
            isopen = False
            print("key, pressed")


    def key_release(k, mod):
        global style_state
        global step
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0
        if k == key.M:
            if style_state == StyleStates.STYLE_1:
                style_state = StyleStates.STYLE_2
            else:
                style_state = StyleStates.STYLE_1


        if k == key.N:
            style_state = StyleStates.SIMULATED
        if k == key.S:
            if style_state == StyleStates.STYLE_1:
                style_state = StyleStates.SPLIT_1
            elif style_state == StyleStates.STYLE_2:
                style_state = StyleStates.SPLIT_2
            elif style_state == StyleStates.SPLIT_1:
                style_state = StyleStates.STYLE_1
            elif style_state == StyleStates.SPLIT_2:
                    style_state = StyleStates.STYLE_2

        if k == key.R:
            style_state = StyleStates.MIXED

        if k == key.C:
            roll_ind[0]+=1
        if k== key.L:
            extra_commands[0] = not extra_commands[0]
        if k== key.E:
            extra_commands[1] = not extra_commands[1]
        if k == key.G:
            extra_commands[2] = not extra_commands[2]
        if k == key.H:
            roll_ind[1] += 1
        if k == key.P:
            commands[2] = not commands[2]


    # env.viewer.window.on_key_press = key_press
    # env.viewer.window.on_key_release = key_release
    win.on_key_press = key_press
    win.on_key_release = key_release
    step = 0
    interpolate_dir = 0

    repetition = 0
    path_list = []
    class_tar = None
    old_class = None
    old_class_tar = None
    while commands[0]:  # isopen:
        now = str(datetime.now())
        env.reset()
        r_history = []  # np.array([-100])
        frames = []  # env.reset()[None,:,:,:]
        road_frames = []  # env.reset()[None,:,:,:]
        input_history = []
        total_reward = 0.0
        steps = 0
        commands[1] = False  # restart = False
        break_count = 0
        left_count = 0
        right_count = 0
        accel_count = 0
        command = np.array([3])
        actual_class_tar = [None, None]
        while True:
            if not commands[2]:
                start_time=time.time()
                curr_classif = CLASSIFIER_LIST[roll_ind[0]% CLASSIFIER_LIST.shape[0]]
                if curr_classif[0] is not None:
                    print("AD")
                    if curr_classif[0] != old_class:
                        print(f"Loading {curr_classif[0]}")
                        classifier = torch.load(curr_classif[0]).to(device)
                        old_class = curr_classif[0]
                        class_tar_list = np.array([[classifier.conv1, "conv1"],[classifier.conv2, "conv2"],[classifier.conv3, "conv3"],[classifier.conv4, "conv4"],[classifier.conv5, "conv5"]])
                    actual_class_tar = class_tar_list[roll_ind[1] % class_tar_list.shape[0]]
                    if actual_class_tar[0] != old_class_tar:
                        cam = GradCAM(model=classifier, target_layer=actual_class_tar[0], use_cuda=True)
                        old_class_tar = actual_class_tar[0]


                    a = np.array([0.0, 0.0, 0.0])

                    if accel_count == 0:
                        if steps == 0:
                            a = np.array([0.0, 1.0, 0.0])
                            start = False
                        else:
                            if command == 0:
                                a[0] = -1.0  # -1.0
                                break_count = 0
                                left_count += 1
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
                else:
                    if curr_classif[0] != old_class:
                        print("Reseting")
                        a =  np.array([0.0, 0.0, 0.0])
                        old_class = curr_classif[0]



                    if actual_class_tar[0] != old_class_tar:
                        old_class_tar = actual_class_tar[0]
                        actual_class_tar = None


                frame, r, done, info, car_frame,broke = env.step(a)
                input_history.append(a)  # =np.vstack((input_history,a))

                total_reward += r

                r_history.append(total_reward)  # = np.append(r_history,r)
                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1

                # landscape-tree-water = env.render(mode="state_pixels", draw_car=False)[:-12, :]
                # road_frames.append(landscape-tree-water)

                # frame=cv2.resize(frame, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
                #frames.append(frame)  # frames = np.vstack((frames,frame[None,:,:,:]))
                # env.render()
                new_frame = np.copy(frame)
                status_bar = new_frame[257:]
                new_frame = new_frame[:256, :256]
                car_frame = np.copy(car_frame)[:256, :256]

                #new_frame = new_frame + 0.0001 * np.random.normal(0, 1, (new_frame.shape[0],new_frame.shape[1], 3))

                if style_state == StyleStates.STYLE_1:
                    new_frame = generate(model, new_frame, a=0,b=1)
                elif style_state == StyleStates.STYLE_2:
                    new_frame = generate(model, new_frame, a=1, b=0)
                elif style_state == StyleStates.SPLIT_1:
                    gen_frame = generate(model, new_frame, a=0, b=1)
                    new_frame = np.stack([np.tril(gen_frame[:,:,i], -1) + np.triu(new_frame[:, :, i]) for i in range(3)],
                                         axis=-1)
                elif style_state == StyleStates.SPLIT_2:
                    gen_frame = generate(model, new_frame, a=1, b=0)
                    new_frame = np.stack([np.tril(gen_frame[:,:,i], -1) + np.triu(new_frame[:, :, i]) for i in range(3)],
                                         axis=-1)
                elif style_state == StyleStates.MIXED:
                    step = step + 1

                    st[interpolate_dir] = 1 - step / 100
                    st[1 - interpolate_dir] = step / 100
                    if step >= 100:
                        step = 0
                        interpolate_dir = 1 - interpolate_dir

                    new_frame = generate(model, new_frame, a=st[0], b=st[1])


                new_frame = np.vstack((new_frame, status_bar))

                new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [204, 0, 0], axis=2) == 3)] = [204, 0, 0]
                new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [76, 76, 76], axis=2) == 3)] = [76, 76, 76]
                new_frame[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [0, 0, 0], axis=2) == 3)] = [0, 0, 0]

                if curr_classif[0] is not None:
                    # Classifer
                    aux_frame = np.copy(new_frame)
                    aux_frame = np.transpose(aux_frame[:256, :, :], axes=[2, 0, 1])
                    aux_frame = (2 * (aux_frame / 255)) - 1
                    aux_frame = aux_frame[np.newaxis, :]
                    x = torch.from_numpy(aux_frame)
                    x = x.to(device=device, dtype=torch.float)

                    if extra_commands[2]:
                        new_frame = np.copy(new_frame)
                        status_bar = new_frame[257:]
                        new_frame = new_frame[:256, :256]/255

                        gradCAm= cam(input_tensor=x, target_category=None)
                        vis =show_cam_on_image(new_frame, gradCAm[0], use_rgb=True)
                        new_frame = np.vstack((vis*255, status_bar))

                    if accel_count == 0:
                        pred = classifier(x)
                        command = pred.argmax(axis=1).cpu().numpy()

                fps = 1.0 / (time.time() - start_time)
                while fps > 61:
                    fps = 1.0 / (time.time() - start_time)


            if extra_commands[0]:
                        #show_stuff(win, frame, width, height, total_reward, fps, car, car_already_drown, steps, command=None, show_help=False, discrete=False, classf_name=None, save=False, helper=None)
                frame = show_stuff(win, np.copy(new_frame), width, height, total_reward, fps, car, True, steps, style_state, command=a,show_help=True,  classf_name = curr_classif[1], save=True, helper =extra_commands, gcam_target=actual_class_tar[1], comms = commands )
                frames.append(frame)
            else:
                show_stuff(win, np.copy(new_frame), width, height, total_reward, fps, car, True, steps, style_state, command=a, show_help=True, classf_name=curr_classif[1], helper =extra_commands, gcam_target=actual_class_tar[1], comms = commands)

            # if done or restart or isopen == False:
            if extra_commands[1]:
                if commands[1] or not commands[0]: #done or broke or
                    break
            else:
                if done or broke or commands[1] or not commands[0]:
                    break
            #  r_history = np.array(r_history)
            # input_history = np.array(input_history)
            # frames = np.array(frames)
            # road_frames = np.array(road_frames)

            # if steps == 1000:
            #     path = create_dir(run_name, repetition, now)
            #     print("Run saved")
            #     path_list.append(path)

    if extra_commands[0]:
        print("Saving")
        import os
        if not os.path.exists(f"Slides/images_from_run/Files/{run_name}"):
            os.makedirs(f"Slides/images_from_run/Files/{run_name}")

        for steps, d in enumerate(frames):
            d.save(f"Slides/images_from_run/Files/{run_name}/step_{steps}.png")
        import subprocess
        os.chdir(f"Slides/images_from_run/Files/{run_name}/")
        subprocess.call(['ffmpeg', '-framerate', '50', '-i', 'step_%1d.png', '-s', f"{width}x{height}", '-vcodec', 'libx265', '-crf', '28', f"../{run_name}.mp4", '-y'])
        os.chdir("..")
        import shutil
        try:
            shutil.rmtree(f"{run_name}")
        except OSError as e:
            print("Error: %s : %s" % ("Images", e.strerror))

        # else:
        #     print("Run not saved")

    env.close()

    # compress_files(win, path_list, width, height)


if __name__ == "__main__":
    main()
