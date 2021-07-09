import pyglet
import cv2
import numpy as np
import os
from copy import deepcopy


def show_stuff(win, frame, width, height, total_reward, fps, car, car_already_drown, steps, style_state, command=None, show_help=False, discrete=False, classf_name=None, save=False, helper=None, gcam_target=None):
    win.switch_to()
    win.dispatch_events()

    if style_state.value == 2 or style_state.value == 4:
        color = (255, 255, 255, 255)
    else:
        color = (0, 0, 0, 255)

    if helper[2]:
        color = (0, 0, 0, 255)

    score_label = pyglet.text.Label(
        "%04i" % total_reward,
        font_size=15,
        x=5,
        y=height * 2.5 / 40.00,
        anchor_x="left",
        anchor_y="center",
        color=(255, 255, 255, 255),
    )

    fps_label = pyglet.text.Label(
        "fps %03i" % fps,
        font_size=16,
        x=width - 200,
        y=height - 26 / 2,
        anchor_x="left",
        anchor_y="center",
        color=color,
    )

    if not car_already_drown:
        frame[np.where(car['frame'][:, :, 0] == 204)] = [204, 0, 0]
        frame[np.where(car['frame'][:256, :, 0] == 0)] = [0, 0, 0]

    frame = cv2.resize(frame, dsize=(width, height), interpolation=cv2.INTER_AREA)

    image_data = np.insert(frame[::-1, :, :], 3, 255, axis=2).flatten().tostring()
    # image_data = ctypes.string_at(id(frame.tostring())+20, 100*100*3)
    image = pyglet.image.ImageData(width=width, height=height, format='RGBA', data=image_data, pitch=width * 4 * 1)
    image.blit(0, 0)

    score_label.draw()
    fps_label.draw()

    help_list = ["Arrows: directions",
                 "esc: exit",
                 "enter: restart",
                 "M: Styletransfer & change style",
                 "N: OpenAI sim",
                 "S: Split screen",
                 "R: Interpolate states",
                 "C: turn AD on and change classifier",
                 "L: Toggle save (log)",
                 "E: Toggle endless mode",
                 "G: Toggle GradCAM",
                 "H: change GradCAM layer"]
    if helper is not None:
        help_list += [f"Save {helper[0]}", f"Endless {helper[1]}"]
    if show_help:
        for i, h in enumerate(help_list, 1):
            help = pyglet.text.Label(
                h,
                font_size=10,
                x=10,
                y=height - 50 - 25 * i,
                anchor_x="left",
                anchor_y="center",
                color=color,
            )
            help.draw()

        # frame[201:231,138:150,0]=128
        # frame[201:231,138:150,1]=0
        # frame[201:231,138:150,2]=128
    if classf_name is not None:
        text = "Classifier: " + classf_name
        classf = pyglet.text.Label(
            text,
            font_size=10,
            x=10,
            y=height - 50 - 25 * (len(help_list) + 1),
            anchor_x="left",
            anchor_y="center",
            color=color,
        )
        classf.draw()

    if gcam_target is not None:
        text = f"gradCAM target: {gcam_target}"
        classf = pyglet.text.Label(
            text,
            font_size=10,
            x=10,
            y=height - 50 - 25 * (len(help_list) + 2),
            anchor_x="left",
            anchor_y="center",
            color=color,
        )
        classf.draw()

    if command is not None:
        if not discrete:
            dirs = []
            if command[0] == 1.0:
                dirs += ["Right"]
            elif command[0] == - 1.0:
                dirs += ["Left"]
            if command[1] > 0:
                dirs += ["Accelerate"]
            if command[2] > 0:
                dirs += ["Brake"]
            if len(dirs) == 0:
                dirs = ['Nothing']
            commands = ', '.join(dirs)
            command_label = pyglet.text.Label(
                commands,
                font_size=16,
                x=10,
                y=height - 26 / 2,
                anchor_x="left",
                anchor_y="center",
                color=color,
            )
            command_label.draw()
        else:
            directions = ["Lef",
                          "Rig",
                          "Acc",
                          "Brk"]
            commands = ', '.join([directions[c[0]] for c in command[-20:][::-2]])
            command_label = pyglet.text.Label(
                commands,
                font_size=16,
                x=10,
                y=height - 26 / 2,
                anchor_x="left",
                anchor_y="center",
                color=color,
            )
            command_label.draw()

    # save_name ="show_diff_styles"
    # if not os.path.exists(f"Slides/images_from_run/{save_name}"):
    #     os.makedirs("Slides/images_from_run/show_diff_styles")
    if save:
        frame = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()

    # event
    win.flip()
    win.clear()

    # ffmpeg - framerate 60 - i   step_% 1d.png  output.mp4

    if save:
        return frame


def create_dir(run_name, repetition, now):
    path = os.getcwd()
    path = os.path.join(path, f"data/runs/{run_name}", now)
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def compress_files(win, path_list, width, height, file_name="histories"):
    win.switch_to()
    win.dispatch_events()
    label = pyglet.text.Label(f"Compressing {len(path_list)} files",
                              font_name='Times New Roman',
                              font_size=36,
                              x=width // 2, y=height // 2,
                              anchor_x='center', anchor_y='center')
    label.draw()
    win.flip()

    win.clear()

    for i, file in enumerate(path_list):
        win.switch_to()
        win.dispatch_events()
        label = pyglet.text.Label(f"Compressing {len(path_list)} files, {int((i / len(path_list)) * 100)} %",
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=width // 2, y=height // 2,
                                  anchor_x='center', anchor_y='center')
        label.draw()
        win.flip()

        win.clear()
        file += "/" + file_name + ".npz"
        un_comp = np.load(file)
        np.savez_compressed(file, r_history=un_comp['r_history'], input_history=un_comp['input_history'], frames=un_comp['frames'], fps=un_comp["fps"], seg_frames=un_comp["seg_frames"], car_frames=un_comp["car_frames"])  # , road_frames=road_frames)

    win.close()
