import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
from numpy.random import default_rng
from skimage import feature
from skimage import measure
from skimage.color import rgb2gray
from skimage.morphology import dilation, closing, white_tophat, disk


def seg_sim(image):
    aux2 = np.zeros((image.shape[0], 256, 256, 1), dtype=np.uint8)
    aux2[np.where(image[:, :256, :, 1] == 105)] = [255]
    aux2[np.where(image[:, :256, :, 1] == 107)] = [255]
    aux2[np.where(image[:, :256, :, 1] == 102)] = [255]
    return aux2


def do_canny_single(img, label, img_type):
    if img_type == 'real':
        img_gray = rgb2gray(img)
        if label == 1:
            # brown
            img_gray[img_gray <= 0.5] = 0
            img_gray[img_gray > 0.5] = 1
        else:
            # grey
            img_gray = feature.canny(img_gray, sigma=6, low_threshold=0.1)

        edges = img_gray  # pointer magic
    else:
        img_gray = np.squeeze(seg_sim(img[None, :, :, :]))
        edges = feature.canny(img_gray, sigma=3, low_threshold=0.2)
    return edges


def curvature(image, label, img_type="real"):
    """
    returns: 1:left
             2:straight
             3:right
             4:no road
             5:undefined
    """
    # Classic straight-line Hough transform

    edge = do_canny_single(image, label, img_type)

    if img_type == "real":
        closed = closing(edge, disk(20))

        wt = white_tophat(closed, disk(1))

    else:
        wt = edge

    r = dilation(wt, disk(2))
    r = np.pad(r, 2)
    contours = measure.find_contours(r, 0.8)

    road_type = 5
    defined = True
    if len(contours) != 0:
        contours = contours[np.argmax([c.shape[0] for c in contours])]

        x = np.vstack(contours)[:, 0]
        y = np.vstack(contours)[:, 1]

        mymodel = np.poly1d(np.polyfit(x, y, 2))

        if mymodel(256) > 300 or mymodel(256) < -50:
            defined = False
    else:
        defined = False
        road_type = 4

    bottom = np.sum(edge[-20:, :])
    if bottom > 10 and defined:
        d2 = np.polyder(mymodel, m=2)
        d2 = d2(0) * 1000
        if d2 < -1:
            road_type = 1
        elif d2 > 1:
            road_type = 3
        else:
            road_type = 2
    return road_type


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
                 '%d' % int(height),
                 ha='center', va='bottom')


def plot_styles(x_0, x_1, saved_frame):
    f, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].imshow(x_0[0], cmap=cm.gray)
    ax[0].set_title('Style 0')
    ax[0].set_axis_off()

    ax[1].imshow(x_1[0], cmap=cm.gray)
    ax[1].set_title('Style 1')
    ax[1].set_axis_off()

    ax[2].imshow(saved_frame[0], cmap=cm.gray)
    ax[2].set_title('OpenAI')
    ax[2].set_axis_off()
    plt.show()


def plot_histos(x_0_curv, x_1_curv, OpenAI_curv, x_0, x_1, saved_frame, name):
    fig = plt.figure(figsize=(20, 10))
    X = np.arange(5)

    plt.title(
        f"Dataset target distribution for: \nStyle 0 dataset {x_0.shape[0]} examples,\nStyle 1 dataset {x_1.shape[0]} examples,\nOpenAI dataset {saved_frame.shape[0]} examples")

    x_0_unique = np.unique(x_0_curv, return_counts=True)[1]
    while x_0_unique.shape[0] < 5:  # Add 0 to no road and undefined
        x_0_unique = np.hstack((x_0_unique, 0))

    x_1_unique = np.unique(x_1_curv, return_counts=True)[1]

    while x_1_unique.shape[0] < 5:  # Add 0 to no road and undefined
        x_1_unique = np.hstack((x_1_unique, 0))

    x_0_rect = plt.bar(X + 0.00, x_0_unique, color='b', width=0.25)
    x_1_rect = plt.bar(X + 0.25, x_1_unique, color='g', width=0.25)

    ai_unique = np.unique(OpenAI_curv, return_counts=True)[1]

    while ai_unique.shape[0] < 5:  # Add 0 to no road and undefined
        ai_unique = np.hstack((ai_unique, 0))

    AI_rect = plt.bar(X + 0.5, ai_unique, color='orange', width=0.25)

    plt.legend(labels=['style 0', 'style 1', 'OpenAI'])
    autolabel(x_0_rect)
    autolabel(x_1_rect)
    autolabel(AI_rect)

    plt.xticks(X, ["left", "straight", "right", "no road", "undefined"])
    plt.savefig(f"./Slides/Curves/{name}")
    plt.show()


def save_curves(curves, dataset, clas, counter, path, size, img_type, seed=42):
    rng = default_rng(seed=seed)
    for curve_type in [1, 3]:
        curves = np.array(curves)
        sel_curves = dataset[np.where(curves == curve_type)]
        if sel_curves.shape[0] < size:
            sel_curves = np.vstack((sel_curves, sel_curves[
                rng.choice(sel_curves.shape[0], size - sel_curves.shape[0])]))  # repeat random curves until complition

        indexes = rng.choice(sel_curves.shape[0], size=size, replace=False)
        for image in sel_curves[indexes]:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if img_type == "real":
                cv2.imwrite(f"{path}/patch_{counter}_{clas}.jpg", image)
                cv2.imwrite(f"{path}/patch_{counter + 1}_{clas}.jpg", cv2.flip(image, 1))
            else:
                cv2.imwrite(f"{path}/patch_{counter}.jpg", image)
                cv2.imwrite(f"{path}/patch_{counter + 1}.jpg", cv2.flip(image, 1))
            counter += 2
    return counter


def save_straights(curves, dataset, clas, counter, path, size, img_type, seed=42):
    rng = default_rng(seed=seed)
    curves = np.array(curves)
    sel_curves = dataset[np.where(curves == 2)]
    if sel_curves.shape[0] < size:
        sel_curves = np.vstack((sel_curves, sel_curves[
            rng.choice(sel_curves.shape[0], size - sel_curves.shape[0])]))  # repeat random curves until completion

    indexes = rng.choice(sel_curves.shape[0], size=size, replace=False)
    for image in sel_curves[indexes]:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if img_type == "real":
            cv2.imwrite(f"{path}/patch_{counter}_{clas}.jpg", image)
        else:
            cv2.imwrite(f"{path}/patch_{counter}.jpg", image)
        counter += 1
    return counter


def save_rest(curves, dataset, clas, counter, path, size, img_type, seed=42):
    rng = default_rng(seed=seed)
    curves = np.array(curves)
    sel_curves = dataset[np.where((curves == 4) | (curves == 5))]
    indexes = rng.choice(sel_curves.shape[0], size=(sel_curves.shape[0] if sel_curves.shape[0] < size else size),
                         replace=False)
    for image in sel_curves[indexes]:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if img_type == "real":
            cv2.imwrite(f"{path}/patch_{counter}_{clas}.jpg", image)
        else:
            cv2.imwrite(f"{path}/patch_{counter}.jpg", image)

        counter += 1
    return counter


def save_to_dataset(curves, dataset, clas, path, conf, counter=0):
    try:
        # Create target Directory
        os.makedirs(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")

    size = conf[0]
    rest = conf[1]
    img_type = conf[2]

    counter = save_curves(curves, dataset, clas, counter, path, int(size / 2), img_type)
    counter = save_straights(curves, dataset, clas, counter, path, size, img_type)
    save_rest(curves, dataset, clas, counter, path, rest, img_type)


def get_seg_data(target_dataset_name, real_image_0, real_image_1, OpenAI_runs):
    class0 = glob.glob(f'patches/{real_image_0}/*')
    class1 = glob.glob(f'patches/{real_image_1}/*')

    dataset_path_OpenAI_seg = f"./styletransfer/cut/datasets/{target_dataset_name}/trainSegA"
    x_0 = np.array([np.asarray(Image.open(fname)) for fname in class0])
    x_1 = np.array([np.asarray(Image.open(fname)) for fname in class1])

    saved_data = np.load(f"runs/{OpenAI_runs[0]}/discrete_histories.npz")
    saved_frame_seg = saved_data["seg_frames"][:, :256, :, :]

    for run in OpenAI_runs[1:]:
        saved_data = np.load(f"runs/{run}/discrete_histories.npz")
        saved_frame_seg = np.append(saved_frame_seg, saved_data["seg_frames"][:, :256, :, :], axis=0)

    dataset_path_seg = f"./styletransfer/cut/datasets/{target_dataset_name}/trainSegB"

    return dataset_path_seg, dataset_path_OpenAI_seg, x_0, x_1, saved_frame_seg


def get_data(target_dataset_name, real_image_0, real_image_1, OpenAI_runs):
    dataset_path_OpenAI = f"./styletransfer/cut/datasets/{target_dataset_name}/trainA"
    dataset_path_real = f"./styletransfer/cut/datasets/{target_dataset_name}/trainB"

    class0 = glob.glob(f'patches/{real_image_0}/*')
    class1 = glob.glob(f'patches/{real_image_1}/*')

    x_0 = np.array([np.asarray(Image.open(fname)) for fname in class0])
    x_1 = np.array([np.asarray(Image.open(fname)) for fname in class1])

    saved_data = np.load(f"runs/{OpenAI_runs[0]}/discrete_histories.npz")
    saved_frame = saved_data["frames"][:, :256, :, :]

    for run in OpenAI_runs[1:]:
        saved_data = np.load(f"runs/{run}/discrete_histories.npz")
        saved_frame = np.append(saved_frame, saved_data["frames"][:, :256, :, :], axis=0)

    return dataset_path_OpenAI, dataset_path_real, x_0, x_1, saved_frame


def get_curves(x_0, x_1, saved_frame):
    x_0_curv = [curvature(img, label) for img, label in zip(x_0, np.zeros(x_0.shape[0]))]
    x_1_curv = [curvature(img, label) for img, label in zip(x_1, np.ones(x_1.shape[0]))]
    OpenAI_curv = [curvature(img, None, img_type="OpenAI") for img in saved_frame]

    return x_0_curv, x_1_curv, OpenAI_curv
