#!/usr/bin/env python
import car_racing


from datetime import datetime
import numpy as np
from pyglet.window import key, Window
from render_utils import show_stuff, create_dir, compress_files
import time

from styletransfer.predict import prepare, generate
from glob import glob
import time_limit

def main():
    style_transf = False
    car = np.load("simulation/car_positions.npz")
    env = car_racing.CarRacing()#gym.make('CarRacing-v0')

    env = time_limit.TimeLimit(env, max_episode_steps=1000)


    if style_transf:
        model = prepare(name="dual_color_hist_tree", mode="FastCUT")
    run_name = "For plotting"

    existing_files = len(glob(f'data/runs/{run_name}/**/*.npz'))
    r_history = []  # np.array([-100])
    frames = []  # env.reset()[None,:,:,:]
    road_frames = []  # env.reset()[None,:,:,:]
    env.reset()

    width = 1000
    height = 800
    win = Window(width=width, height=height)

    commands = [True, False, 3, False]  # isOpen, restert, [1 Left, 2 Right, 3 accelerate, 4 stop], Pause
    # isopen, restart
    # isopen = True
    # restart = False

    a = np.array([0.0, 0.0, 0.0])  # isOpen, restert, Left, Right, accelerate, stop
    input_history = []  # np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        # global restart
        # global isopen
        if k == 0xFF0D:
            restart = True
            commands[1] = True
        if k == key.LEFT:
            # a[0] = -1.0
            commands[2] = 1
        elif k == key.RIGHT:
            commands[2] = 2
        elif k == key.UP:
            commands[2] = 3
        elif k == key.DOWN:
            commands[2] = 4
        else:
            commands[2] = 3
        if k == key.P:
            commands[3] = not commands[3]

        if k == key.ESCAPE:
            commands[0] = False
            isopen = False

    def key_release(k, mod):
        if k == key.LEFT:
            if commands[2]==1:
                commands[2] = 3

        if k == key.RIGHT:
            if commands[2]==2:
                commands[2] = 3

        if k == key.UP:
            if commands[2]==3:
                commands[2] = 3
        if k == key.DOWN:
            if commands[2]==4:
                commands[2] = 3

    # env.viewer.window.on_key_press = key_press
    # env.viewer.window.on_key_release = key_release
    win.on_key_press = key_press
    win.on_key_release = key_release

    repetition = 0
    path_list = []
    while commands[0]:  # isopen:

        env.reset()
        r_history = []  # np.array([-100])
        frames = []  # env.reset()[None,:,:,:]
        style_0=[]
        style_1=[]
        seg_frames = []
        car_frames =[]

        road_frames = []  # env.reset()[None,:,:,:]
        input_history = []
        total_reward = 0.0
        steps = 0
        commands[1] = False  # restart = False
        s = [0, 1]
        save = False
        while True:

            start_time = time.time()
            a = np.array([0.0, 0, 0.0])
            if commands[2] == 1:
                a[0] = -1.0
            elif commands[2] == 2:
                a[0] = +1.0
            elif commands[2] == 3:
                a[1] = +1.0 # +0.9
            elif commands[2] == 4:
                a[2] = +0.5
            else:
                print("Wrong input")

            if not commands[3]:
                frame_seg = env.render_segmentation()
                if steps>0:
                    input_history.append(commands[2])
                frame, r, done, info, car_frame, broke = env.step(a)
                 # =np.vstack((input_history,a))
                total_reward += r
                r_history.append(total_reward)  # = np.append(r_history,r)

                if steps % 200 == 0 or done:
                    print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                steps += 1

                # frame=cv2.resize(frame, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
                frames.append(frame)  # frames = np.vstack((frames,frame[None,:,:,:]))
                seg_frames.append(frame_seg)
                car_frames.append(car_frame)
                # env.render()
                new_frame = np.copy(frame)

                if style_transf:
                    status_bar = np.copy(new_frame[257:])
                    new_frame = np.copy(new_frame[:256, :256])
                    new_frame_0 = generate(model, new_frame, 1, 0)
                    new_frame_1 = generate(model, new_frame, 0, 1)
                    new_frame_0 = np.vstack((new_frame_0, status_bar))
                    new_frame_1 = np.vstack((new_frame_1, status_bar))

                    # new_frame_0[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [204, 0, 0], axis=2) == 3)] = [204, 0, 0]
                    # new_frame_0[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [76, 76, 76], axis=2) == 3)] = [76, 76, 76]
                    # new_frame_0[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [0, 0, 0], axis=2) == 3)] = [0, 0, 0]
                    style_0.append(new_frame_0)

                    # new_frame_1[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [204, 0, 0], axis=2) == 3)] = [204, 0, 0]
                    # new_frame_1[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [76, 76, 76], axis=2) == 3)] = [76, 76, 76]
                    # new_frame_1[201:230, 120:136][np.where(np.sum(car_frame[201:230, 120:136] == [0, 0, 0], axis=2) == 3)] = [0, 0, 0]
                    style_1.append(new_frame_1)


                new_frame[201:230,120:136][np.where(np.sum(car_frame[201:230,120:136] == [204, 0, 0], axis=2) == 3)] = [204, 0, 0]
                new_frame[201:230,120:136][np.where(np.sum(car_frame[201:230,120:136] == [76, 76, 76], axis=2) == 3)] = [76, 76, 76]
                new_frame[201:230,120:136][np.where(np.sum(car_frame[201:230,120:136] == [0, 0, 0], axis=2) == 3)] = [0, 0, 0]

                fps = 1.0 / (time.time() - start_time)
                while fps > 61:
                    fps = 1.0 / (time.time() - start_time)

            if steps == 1:
                cmd= None
            else:
                cmd = [np.array([c-1]) for c in input_history]

            show_stuff(win, np.copy(new_frame), width, height, total_reward, fps, car, True, steps, cmd, discrete=True)

            # if done or restart or isopen == False:
            if done:
                save = True
                break
            if broke or commands[1] or not commands[0]:

                break
        r_history = np.array(r_history)
        input_history = np.array(input_history)
        frames = np.array(frames)
        seg_frames = np.array(seg_frames)
        car_frames = np.array(car_frames)
        style_0 = np.array(style_0)
        style_1 = np.array(style_1)

        # road_frames = np.array(road_frames)
        #if steps == 1000:
        if save:
            now = str(datetime.now())
            path = create_dir(run_name, repetition, now)
            path_list.append(path)
            print(f"Run saved (in list {len(path_list)} saved and compressed {existing_files} in total ({existing_files+len(path_list)})")
            if style_transf:
                np.savez(f'{path}/style_transf_histories', r_history=r_history, input_history=input_history, frames=frames[:-1], fps=fps, seg_frames=seg_frames[:-1], car_frames=car_frames[:-1], style_0=style_0[:-1], style_1=style_1[:-1] )  # , road_frames=road_frames)
            else:
                np.savez(f'{path}/discrete_histories', r_history=r_history, input_history=input_history, seg_frames=seg_frames[:-1], fps=fps, frames=frames[:-1], car_frames=car_frames[:-1])  # , road_frames=road_frames)
        else:
            print(f"Run NOT saved (in list {len(path_list)} saved and compressed {existing_files} in total ({existing_files+len(path_list)})")


        if 100 - (existing_files + len(path_list)) == 0:
            commands[0] =False

    env.close()

    if not style_transf:
        compress_files(win, path_list, width, height, file_name="discrete_histories")

    print(f"Saving {len(path_list)} runs, in {run_name} are {len(glob(f'data/runs/{run_name}/**/*.npz'))} runs")


if __name__ == "__main__":
    main()
