import numpy as np
import torch

from styletransfer.cut.models import create_model
from styletransfer.cut.options.test_options import TestOptions


def prepare(name="dual_simple", mode="FastCUT" , path = "styletransfer/cut/checkpoints"):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for testmodel
    opt.name = name  # "road_FastCUT" # NAME "road_CUT" #
    opt.checkpoints_dir = path
    opt.CUT_mode = mode  # "FastCUT" # CUT
    opt.phase = "test"
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = "conditional"
    opt.netG = "conditional_resnet_9"
    opt.netD = "conditional"
    opt.model = "conditional_cut"
    opt.lambda_hist = 0
    opt.lambda_edge = 0
    opt.edge_warmup=0

    model = create_model(opt)  # create a model given opt.model and other options

    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    return model


def generate(model, img, a, b):
    img = np.transpose(img, axes=[2, 0, 1])
    img = img[np.newaxis, :]
    img = 2 * ((img / 255) - 1)
    data = {
        "A": torch.Tensor(img),
        "B": torch.Tensor(img),
        "B_class": torch.Tensor([[a, b]]),
        "A_paths": "None",
        "B_paths": "None"
    }
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image result# s
    img = visuals['fake_B'].cpu().numpy()
    from matplotlib import pyplot as plt

    img = np.transpose(img, axes=[0, 2, 3, 1])

    img = ((np.reshape(img, (256, 256, 3)) + 1) / 2) * 255
    img = img.astype(np.uint8)
    return img

def generate_no_conv(model, img, a, b):
    img = img[np.newaxis, :]
    data = {
        "A": torch.Tensor(img),
        "B": torch.Tensor(img),
        "B_class": torch.Tensor([[a, b]]),
        "A_paths": "None",
        "B_paths": "None"
    }
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image result# s
    img = visuals['fake_B'].cpu().numpy()
    from matplotlib import pyplot as plt

    return img

def prepare_single_style(name="road_FastCUT", mode="FastCUT"):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for testmodel
    opt.name = name  # "road_FastCUT" # NAME "road_CUT" #
    opt.checkpoints_dir = "styletransfer/cut/checkpoints"
    opt.model = mode  # "FastCUT" # CUT
    opt.phase = "test"
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.dataset_mode = "single"
    model = create_model(opt)  # create a model given opt.model and other options

    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    return model


def generate_single_style(model, img):
    img = np.transpose(img, axes=[2, 0, 1])
    img = img[np.newaxis, :]
    data = {
        "A": torch.Tensor(img),
        "B": torch.Tensor(img),
        "A_paths": "None",
        "B_paths": "None"
    }
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image result# s
    img = visuals['fake_B'].cpu().numpy()
    img = np.transpose(img, axes=[0, 2, 3, 1])
    img = ((np.reshape(img, (256, 256, 3)) + 1) / 2) * 255
    img = img.astype(np.uint8)
    return img


def prepare_edges_only(name, loss, mode="edge"):
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for testmodel
    opt.name = name  # "road_FastCUT" # NAME "road_CUT" #
    opt.checkpoints_dir = "styletransfer/cut/checkpoints"
    opt.model = mode  # "FastCUT" # CUT
    opt.phase = "test"
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.edge_loss = loss
    opt.dataset_mode = "conditional"
    opt.output_nc = 1
    opt.ngf = 16
    model = create_model(opt)  # create a model given opt.model and other options

    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()
    return model


def generate_edges_only(model, img, a, b):
    img = np.transpose(img, axes=[2, 0, 1])
    img = img[np.newaxis, :]
    img = 2 * ((img / 255) - 1)
    data = {
        "A": torch.Tensor(img),
        "B": torch.Tensor(img),
        "B_class": torch.Tensor([[a, b]]),
        "A_paths": "None",
        "B_paths": "None"
    }
    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference
    visuals = model.get_current_visuals()  # get image result# s
    img = visuals['output'].cpu().numpy()
    img = np.transpose(img, axes=[0, 2, 3, 1])
    img = ((np.reshape(img, (256, 256, 1)) + 1) / 2) * 255
    img = img.astype(np.uint8)
    return img
