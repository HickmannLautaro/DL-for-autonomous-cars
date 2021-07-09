import glob
import sys
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import SubplotSpec
from skimage import feature, measure
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.morphology import dilation, closing, white_tophat, disk
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

warnings.filterwarnings("ignore",category=UserWarning)

frames = 50
frame_steps = 5

def seg_sim(image):
    aux2 = np.zeros((256, 256, 1), dtype=np.uint8)
    aux2[np.where((image[:256, :, 1] >= 98) & (image[:256, :, 1] <= 112))] = [255]
    aux2[np.where(image[:256, :, 1] == 107)] = [255]
    aux2[np.where(image[:256, :, 1] == 102)] = [255]
    aux2[np.where(image[:256, :, 1] == 111)] = [255]

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
        img_gray = np.squeeze(seg_sim(img))
        edges = feature.canny(img_gray, sigma=3, low_threshold=0.2)
    return edges


def MSE(YH, Y):
    loss = torch.nn.MSELoss()
    return loss(torch.from_numpy(Y), torch.from_numpy(YH)).numpy()


def BCE(YH, Y):
    loss = torch.nn.BCELoss()
    return loss(torch.from_numpy(Y), torch.from_numpy(YH)).numpy()


def histo_single_grayscale(image, title, ax):
    img = [image]
    img_gray = np.array([cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in img])
    hist = cv2.calcHist(img_gray, [0], None, [256], [0, 256])
    ax.cla()
    ax.set_title(f"Grayscale Histogram {title}")
    ax.set_xlabel("Bins")
    ax.set_ylabel("# of Pixels")
    ax.plot(hist)
    ax.set_xlim([0, 256])
    return ax


def histo_single(image, title, ax):
    img = [image]
    colors = ("r", "g", "b")
    ax.cla()
    ax.set_title(f"'Flattened' Color Histogram {title}")
    ax.set_xlabel("Bins")
    ax.set_ylabel("# of Pixels")
    for channel in range(3):
        hist = cv2.calcHist(img, [channel], None, [256], [0, 256])
        ax.plot(hist, color=colors[channel])
        ax.set_xlim([0, 256])
    return ax


def real_vs_gan_histo(real_data, gan_data, style):
    fig, ax = plt.subplots(2, 5, figsize=(30, 10))
    n = 0
    fig.suptitle(f"Histogram comparison for style {style}, frame {n}")
    ax[0, 0].imshow(real_data[n])
    ax[0, 0].set_title('Input real image sample')
    ax[0, 0].set_axis_off()

    ax[0, 1] = histo_single_grayscale(real_data[n], f"single image Real", ax[0, 1])
    ax[0, 2] = histo_single_grayscale(real_data.reshape(real_data.shape[0] * real_data.shape[1], real_data.shape[2], real_data.shape[3]), f"mean Real", ax[0, 2])
    ax[0, 3] = histo_single(real_data[n], f"single image Real", ax[0, 3])
    ax[0, 4] = histo_single(real_data.reshape(real_data.shape[0] * real_data.shape[1], real_data.shape[2], real_data.shape[3]), f"mean Real", ax[0, 4])

    ax[1, 0].imshow(gan_data[n])
    ax[1, 0].set_title('Generated image sample')
    ax[1, 0].set_axis_off()

    ax[1, 1] = histo_single_grayscale(gan_data[n], f"single image Generated", ax[1, 1])
    ax[1, 2] = histo_single_grayscale(gan_data.reshape(gan_data.shape[0] * gan_data.shape[1], gan_data.shape[2], gan_data.shape[3]), f"mean Generated", ax[1, 2])
    ax[1, 3] = histo_single(gan_data[n], f"single image Generated", ax[1, 3])
    ax[1, 4] = histo_single(gan_data.reshape(gan_data.shape[0] * gan_data.shape[1], gan_data.shape[2], gan_data.shape[3]), f"mean Generated", ax[1, 4])
    plt.tight_layout()

    # plt.savefig(f"Slides/histo_style_{style}.png" , bbox_inches='tight')
    def update(n):
        fig.suptitle(f"Histogram comparison for style {style}, frame {n}")
        ax[0, 0].imshow(real_data[n])
        ax[0, 0].set_title('Input real image sample')
        ax[0, 0].set_axis_off()

        ax[0, 1] = histo_single_grayscale(real_data[n], f"single image Real", ax[0, 1])
        ax[0, 3] = histo_single(real_data[n], f"single image Real", ax[0, 3])

        ax[1, 0].imshow(gan_data[n])
        ax[1, 0].set_title('Generated image sample')
        ax[1, 0].set_axis_off()

        ax[1, 1] = histo_single_grayscale(gan_data[n], f"single image Generated", ax[1, 1])
        ax[1, 3] = histo_single(gan_data[n], f"single image Generated", ax[1, 3])

    anim = FuncAnimation(fig, update, frames=frames, interval=1000, save_count=sys.maxsize)
    #anim.save(f'Slides/histo_style_{style}.gif', dpi=150, writer='imagemagick')
    plt.show()
    print(f'Slides/histo_style_{style}.gif done')
    return anim

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def curve_detection(openAI_data, real_style_0, real_style_1):
    def h_trasn_simple(image, label, number, img_type, row):
        # Classic straight-line Hough transform

        edge = do_canny_single(image, label, img_type)

        # Generating figure 1

        ax[row, 0].imshow(image, cmap=cm.gray)
        ax[row, 0].set_title('Input image')
        ax[row, 0].set_axis_off()

        ax[row, 1].imshow(edge, cmap=cm.gray)
        ax[row, 1].set_title('Canny edge detection')
        ax[row, 1].set_axis_off()

        if img_type == "real":
            closed = closing(edge, disk(20))

            wt = white_tophat(closed, disk(1))
        else:
            wt = edge

        r = dilation(wt, disk(2))
        r = np.pad(r, 2)
        contours = measure.find_contours(r, 0.8)

        myline = np.linspace(0, 256, 100)

        ax[row, 2].cla()
        ax[row, 2].imshow(r, cmap=cm.gray)
        if img_type == "real":
            ax[row, 2].set_title('Processed (closing + white_tophat + dilation + padding + contours)')
        else:
            ax[row, 2].set_title('Processed (dilation + padding + contours)')
        ax[row, 2].set_axis_off()

        for contour in contours:
            ax[row, 2].plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax[row, 3].cla()
        ax[row, 3].imshow(np.zeros(edge.shape), cmap=cm.gray)
        ax[row, 3].set_title('Fitted curve and points')
        ax[row, 3].set_axis_off()

        road_type = "Undefined"

        defined = True

        if len(contours) != 0:
            contours = contours[np.argmax([c.shape[0] for c in contours])]

            x = np.vstack(contours)[:, 0]
            y = np.vstack(contours)[:, 1]

            ax[row, 3].scatter(y, x)

            mymodel = np.poly1d(np.polyfit(x, y, 2))
            line = mymodel(myline)

            if mymodel(256) > 300 or mymodel(256) < -50:
                defined = False

            ax[row, 3].plot(line, myline, "--r")
        else:
            defined = False
            road_type = "No road"
        ax[row, 3].set_ylim((edge.shape[0], 0))
        ax[row, 3].set_xlim((0, edge.shape[1]))

        bottom = np.sum(edge[-20:, :])
        if bottom > 10 and defined:
            d2 = np.polyder(mymodel, m=2)
            d2 = d2(0) * 1000
            if d2 < -1:
                road_type = "left curve"
            elif d2 > 1:
                road_type = "right curve"
            else:
                road_type = "Straight road"

            road_type = f"{road_type}\nd2: {d2}"

        create_subtitle(fig, grid[row, ::], f"{road_type}")

    n = 0
    fig, ax = plt.subplots(3, 4, figsize=(20, 10))
    grid = plt.GridSpec(3, 4)

    h_trasn_simple(openAI_data[n], None, n, "OpenAI", 0)

    h_trasn_simple(real_style_0[n], 0, n, "real", 1)

    h_trasn_simple(real_style_1[n], 1, n, "real", 2)
    plt.tight_layout()

    def update(n):
        h_trasn_simple(openAI_data[n], None, n, "OpenAI", 0)

        h_trasn_simple(real_style_0[n], 0, n, "real", 1)

        h_trasn_simple(real_style_1[n], 1, n, "real", 2)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000, save_count=sys.maxsize)
    anim.save(f'Slides/curves.gif', dpi=150, writer='imagemagick')
    print(f'Slides/curves.gif done')
    return anim

def edge_comp(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges):
    openAI_data, gan_data_0, gan_data_1 = experiments
    n = 0
    fig, ax = plt.subplots(3, 4, figsize=(20, 10))
    fig.suptitle(f"Structure comparison, frame {n}")

    ax[0, 0].imshow(openAI_data[n])
    ax[0, 0].set_title('Input OpenAI image sample')
    ax[0, 0].set_axis_off()

    ax[1, 0].imshow(gan_data_0[n])
    ax[1, 0].set_title('Generated Style 0')
    ax[1, 0].set_axis_off()

    ax[2, 0].imshow(gan_data_1[n])
    ax[2, 0].set_title('Generated Style 1')
    ax[2, 0].set_axis_off()

    ax[0, 1].imshow(do_canny_single(openAI_data[n], None, "OpenAI"), cmap=cm.gray)
    ax[0, 1].set_title('Input OpenAI image sample')
    ax[0, 1].set_axis_off()

    ax[1, 1].imshow(do_canny_single(gan_data_0[n], 0, "real"), cmap=cm.gray)
    ax[1, 1].set_title('Generated Style 0')
    ax[1, 1].set_axis_off()

    ax[2, 1].imshow(do_canny_single(gan_data_1[n], 1, "real"), cmap=cm.gray)
    ax[2, 1].set_title('Generated Style 1')
    ax[2, 1].set_axis_off()

    ax[0, 2].set_title("Sample Accuracy error")
    ax[0, 2].bar(0, 1 - np.mean(gan_data_0_edges[n] == OpenAI_edges[n]), color='b')
    ax[0, 2].bar(1, 1 - np.mean(gan_data_1_edges[n] == OpenAI_edges[n]), color='g')
    ax[0, 2].set_xticks([0, 1])
    ax[0, 2].set_xticklabels(["style 0", "style 1"])

    ax[1, 2].set_title("Sample MSE")
    ax[1, 2].bar(0, get_losses(gan_data_0_edges[n], y=OpenAI_edges[n])[0], color='b')
    ax[1, 2].bar(1, get_losses(gan_data_1_edges[n], y=OpenAI_edges[n])[0], color='g')
    ax[1, 2].set_xticks([0, 1])
    ax[1, 2].set_xticklabels(["style 0", "style 1"])

    ax[2, 2].set_title("Sample BCE")
    ax[2, 2].bar(0, get_losses(gan_data_0_edges[n], y=OpenAI_edges[n])[1], color='b')
    ax[2, 2].bar(1, get_losses(gan_data_1_edges[n], y=OpenAI_edges[n])[1], color='g')
    ax[2, 2].set_xticks([0, 1])
    ax[2, 2].set_xticklabels(["style 0", "style 1"])

    ax[0, 3].set_title("Overall Accuracy error")
    ax[0, 3].bar(0, 1 - np.mean(gan_data_0_edges == OpenAI_edges), color='b')
    ax[0, 3].bar(1, 1 - np.mean(gan_data_1_edges == OpenAI_edges), color='g')
    ax[0, 3].set_xticks([0, 1])
    ax[0, 3].set_xticklabels(["style 0", "style 1"])

    ax[1, 3].set_title("Overall MSE")
    ax[1, 3].bar(0, get_losses(gan_data_0_edges)[0], color='b')
    ax[1, 3].bar(1, get_losses(gan_data_1_edges)[0], color='g')
    ax[1, 3].set_xticks([0, 1])
    ax[1, 3].set_xticklabels(["style 0", "style 1"])

    ax[2, 3].set_title("Overall BCE")
    ax[2, 3].bar(0, get_losses(gan_data_0_edges)[1], color='b')
    ax[2, 3].bar(1, get_losses(gan_data_1_edges)[1], color='g')
    ax[2, 3].set_xticks([0, 1])
    ax[2, 3].set_xticklabels(["style 0", "style 1"])
    plt.tight_layout()
    def update(n):
        fig.suptitle(f"Structure comparison, frame {n}")
        ax[0, 0].imshow(openAI_data[n])
        ax[0, 0].set_title('Input OpenAI image sample')
        ax[0, 0].set_axis_off()

        ax[1, 0].imshow(gan_data_0[n])
        ax[1, 0].set_title('Generated Style 0')
        ax[1, 0].set_axis_off()

        ax[2, 0].imshow(gan_data_1[n])
        ax[2, 0].set_title('Generated Style 1')
        ax[2, 0].set_axis_off()

        ax[0, 1].imshow(do_canny_single(openAI_data[n], None, "OpenAI"), cmap=cm.gray)
        ax[0, 1].set_title('Input OpenAI image sample')
        ax[0, 1].set_axis_off()

        ax[1, 1].imshow(do_canny_single(gan_data_0[n], 0, "real"), cmap=cm.gray)
        ax[1, 1].set_title('Generated Style 0')
        ax[1, 1].set_axis_off()

        ax[2, 1].imshow(do_canny_single(gan_data_1[n], 1, "real"), cmap=cm.gray)
        ax[2, 1].set_title('Generated Style 1')
        ax[2, 1].set_axis_off()

        ax[0, 2].set_title("Sample Accuracy error")
        ax[0, 2].bar(0, 1 - np.mean(gan_data_0_edges[n] == OpenAI_edges[n]), color='b')
        ax[0, 2].bar(1, 1 - np.mean(gan_data_1_edges[n] == OpenAI_edges[n]), color='g')
        ax[0, 2].set_xticks([0, 1])
        ax[0, 2].set_xticklabels(["style 0", "style 1"])

        ax[1, 2].set_title("Sample MSE")
        ax[1, 2].bar(0, get_losses(gan_data_0_edges[n], y=OpenAI_edges[n])[0], color='b')
        ax[1, 2].bar(1, get_losses(gan_data_1_edges[n], y=OpenAI_edges[n])[0], color='g')
        ax[1, 2].set_xticks([0, 1])
        ax[1, 2].set_xticklabels(["style 0", "style 1"])

        ax[2, 2].set_title("Sample BCE")
        ax[2, 2].bar(0, get_losses(gan_data_0_edges[n], y=OpenAI_edges[n])[1], color='b')
        ax[2, 2].bar(1, get_losses(gan_data_1_edges[n], y=OpenAI_edges[n])[1], color='g')
        ax[2, 2].set_xticks([0, 1])
        ax[2, 2].set_xticklabels(["style 0", "style 1"])

    anim = FuncAnimation(fig, update, frames=frames, interval=1000, save_count=sys.maxsize)
    anim.save('Slides/edges.gif', dpi=150, writer='imagemagick')
    print(f'Slides/edges.gif done')

    return anim


def edge_comp_graphs(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges):
    openAI_data, gan_data_0, gan_data_1 = experiments
    n = 0
    X = range(OpenAI_edges.shape[0])
    fig, ax = plt.subplots(3, 4, figsize=(20, 10))

    losses_0 = np.array([get_losses_new(y, x) for (x, y) in zip(gan_data_0_edges, OpenAI_edges)])
    losses_1 = np.array([get_losses_new(y, x) for (x, y) in zip(gan_data_1_edges, OpenAI_edges)])
    losses_n = ['Accuracy', 'MSE', 'BCE', 'SSIM', 'PSNR', 'Cosine similarity']

    def update(n):
        fig.suptitle(f"Structure comparison, frame {n}")
        ax[0, 0].imshow(openAI_data[n])
        ax[0, 0].set_title('Input OpenAI image sample')
        ax[0, 0].set_axis_off()

        ax[1, 0].imshow(gan_data_0[n])
        ax[1, 0].set_title('Generated Style 0')
        ax[1, 0].set_axis_off()

        ax[2, 0].imshow(gan_data_1[n])
        ax[2, 0].set_title('Generated Style 1')
        ax[2, 0].set_axis_off()

        ax[0, 1].imshow(do_canny_single(openAI_data[n], None, "OpenAI"), cmap=cm.gray)
        ax[0, 1].set_title('Input OpenAI image sample')
        ax[0, 1].set_axis_off()

        ax[1, 1].imshow(do_canny_single(gan_data_0[n], 0, "real"), cmap=cm.gray)
        ax[1, 1].set_title('Generated Style 0')
        ax[1, 1].set_axis_off()

        ax[2, 1].imshow(do_canny_single(gan_data_1[n], 1, "real"), cmap=cm.gray)
        ax[2, 1].set_title('Generated Style 1')
        ax[2, 1].set_axis_off()

        l = 0  # loss
        for l in range(3):
            ax[l, 2].cla()
            ax[l, 2].set_title(losses_n[l])
            ax[l, 2].scatter(n, losses_0[n, l], color='g')
            ax[l, 2].scatter(n, losses_1[n, l], color='brown')
            ax[l, 2].plot(X, losses_0[:, l], color='g')
            ax[l, 2].plot(X, losses_1[:, l], color='brown')
            ax[l, 2].bar(-5, losses_0[:, l].mean(), color='g', label="Style 0")
            ax[l, 2].bar(-3, losses_1[:, l].mean(), color='brown', label="Style 1")
            ax[l, 2].set_ylim(np.min([losses_0[:, l], losses_1[:, l]]) * 0.99, np.max([losses_0[:, l], losses_1[:, l]]) * 1.01)
            ax[l, 2].set_xlabel("Frame")
            ax[l, 2].set_ylabel(losses_n[l])
            ax[l, 2].legend()

        for l in range(3, 6):
            ax[l - 3, 3].cla()
            ax[l - 3, 3].set_title(losses_n[l])
            ax[l - 3, 3].scatter(n, losses_0[n, l], color='g')
            ax[l - 3, 3].scatter(n, losses_1[n, l], color='brown')
            ax[l - 3, 3].plot(X, losses_0[:, l], color='g')
            ax[l - 3, 3].plot(X, losses_1[:, l], color='brown')
            ax[l - 3, 3].bar(-5, losses_0[:, l].mean(), color='g', label="Style 0")
            ax[l - 3, 3].bar(-3, losses_1[:, l].mean(), color='brown', label="Style 1")
            ax[l - 3, 3].set_ylim(np.min([losses_0[:, l], losses_1[:, l]]) * 0.99, np.max([losses_0[:, l], losses_1[:, l]]) * 1.01)
            ax[l - 3, 3].set_xlabel("Frame")
            ax[l - 3, 3].set_ylabel(losses_n[l])
            ax[l - 3, 3].legend()

        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=np.arange(0, OpenAI_edges.shape[0], frame_steps), interval=500, save_count=sys.maxsize)

    anim.save('Slides/edges_complete.gif', dpi=150, writer='imagemagick')
    print(f'Slides/edges_complete.gif done')

    return anim


def edge_comp_box_plots(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges):
    openAI_data, gan_data_0, gan_data_1 = experiments
    n = 0
    X = range(OpenAI_edges.shape[0])
    fig, ax = plt.subplots(3, 4, figsize=(20, 10))

    losses_0 = np.array([get_losses_new(y, x) for (x, y) in zip(gan_data_0_edges, OpenAI_edges)])
    losses_1 = np.array([get_losses_new(y, x) for (x, y) in zip(gan_data_1_edges, OpenAI_edges)])
    losses_n = ['Accuracy', 'MSE', 'BCE', 'SSIM', 'PSNR', 'Cosine similarity']

    def update(n):
        fig.suptitle(f"Structure comparison, frame {n}")
        ax[0, 0].imshow(openAI_data[n])
        ax[0, 0].set_title('Input OpenAI image sample')
        ax[0, 0].set_axis_off()

        ax[1, 0].imshow(gan_data_0[n])
        ax[1, 0].set_title('Generated Style 0')
        ax[1, 0].set_axis_off()

        ax[2, 0].imshow(gan_data_1[n])
        ax[2, 0].set_title('Generated Style 1')
        ax[2, 0].set_axis_off()

        ax[0, 1].imshow(do_canny_single(openAI_data[n], None, "OpenAI"), cmap=cm.gray)
        ax[0, 1].set_title('Input OpenAI image sample')
        ax[0, 1].set_axis_off()

        ax[1, 1].imshow(do_canny_single(gan_data_0[n], 0, "real"), cmap=cm.gray)
        ax[1, 1].set_title('Generated Style 0')
        ax[1, 1].set_axis_off()

        ax[2, 1].imshow(do_canny_single(gan_data_1[n], 1, "real"), cmap=cm.gray)
        ax[2, 1].set_title('Generated Style 1')
        ax[2, 1].set_axis_off()

        l = 0  # loss
        for c in [2, 3]:
            for r in range(3):
                ax[r, c].cla()
                ax[r, c].set_title(losses_n[l])
                ax[r, c].scatter([0, 1], [losses_0[n, l], losses_1[n, l]], color=['g', 'brown'])
                ax[r, c].boxplot([losses_0[:, l], losses_1[:, l]], positions=[0, 1])
                ax[r, c].set_xticks([0, 1])
                ax[r, c].set_xticklabels(["style 0", "style 1"])
                ax[r, c].set_ylabel(losses_n[l])
                l += 1

        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=np.arange(0, OpenAI_edges.shape[0], frame_steps), interval=500, save_count=sys.maxsize)

    anim.save('Slides/edges_complete_boxplot.gif', dpi=150, writer='imagemagick')
    print(f'Slides/edges_complete_boxplot.gif done')

    return anim

dataset = "road6"
real_style_0 = glob.glob(f'./styletransfer/cut/datasets/{dataset}/trainB/*_0.jpg*')
real_style_1 = glob.glob(f'./styletransfer/cut/datasets/{dataset}/trainB/*_1.jpg*')
real_style_0 = np.array([np.asarray(Image.open(fname)) for fname in real_style_0])
real_style_1 = np.array([np.asarray(Image.open(fname)) for fname in real_style_1])
loaded = np.load("files_for_style_comparisson/new_run.npz")
names = loaded["names"]
steps = loaded["steps"]
experiments = loaded["frames"]
openAI_data, gan_data_0, gan_data_1 = experiments
OpenAI_edges = np.array([do_canny_single(img, None, "OpenAI") for img in openAI_data]).astype(float)


def get_losses(YH, y=OpenAI_edges):
    return [MSE(YH, y), BCE(YH, y)]

def get_losses_new(YH, y=OpenAI_edges):
    return [np.mean(YH==y), MSE(YH, y), BCE(YH, y), ssim(YH,y), psnr(y,YH), cos_sim(y, YH).mean()]



gan_data_0_edges = np.array([do_canny_single(img, 0, "real") for img in gan_data_0]).astype(float)
gan_data_1_edges = np.array([do_canny_single(img, 1, "real") for img in gan_data_1]).astype(float)
print('Creating gifs')
#real_vs_gan_histo(real_style_0, experiments[1], "0")
#real_vs_gan_histo(real_style_1, experiments[2], "1")
edge_comp(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges)
edge_comp_box_plots(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges)
edge_comp_graphs(experiments, OpenAI_edges, gan_data_0_edges, gan_data_1_edges)
#curve_detection(experiments[0], real_style_0, real_style_1)
