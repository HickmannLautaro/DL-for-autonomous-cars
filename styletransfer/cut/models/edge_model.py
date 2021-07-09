"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from skimage import feature
from skimage.color import rgb2gray
import pytorch_ssim

from .base_model import BaseModel
from . import networks
import numpy as np


class EdgeModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.add_argument('--lambda_regression', type=float, default=1.0, help='weight for the regression loss')  # You can define new arguments for this model.
            parser.add_argument('--edge_loss', type=str, default='MSE', choices=['MSE', 'BCE', 'SSIM'], help='what loss to use for edge detection comparison')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['MSE', 'invSSIM', 'BCE']  # Test added so that viscom can plot it, for only one loss it breaks
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['data_A', 'data_B', 'output', 'target']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['E']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.edge_loss = opt.edge_loss

        self.netE = networks.define_Seg(in_channels=opt.input_nc, out_channel=opt.output_nc, ngf=opt.ngf, gpu_ids=self.gpu_ids)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.SSIM = pytorch_ssim.SSIM(window_size=11)
            self.BCE_loss = torch.nn.BCELoss()
            self.MSE = torch.nn.MSELoss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def seg_sim(self, image):
        aux2 = np.zeros((image.shape[0], 1, 256, 256), dtype=np.uint8)
        aux2[np.where(image[:, 1, :256, :] == 105)] = [255]
        aux2[np.where(image[:, 1, :256, :] == 107)] = [255]
        aux2[np.where(image[:, 1, :256, :] == 102)] = [255]
        return aux2

    def update_epoch(self, epoch):
        self.epoch = epoch - 1  # -1 to turn off in first epoch

    def do_canny(self, images, simulated=False):

        if not simulated:  # generated image
            images = images.cpu().detach().numpy()
            #    print(images.shape)
            labels = self.y.cpu().numpy()
            img_gray = rgb2gray(np.transpose(images, [0, 2, 3, 1]))

            for i, _ in enumerate(img_gray):

                if np.argmax(labels[i]) == 1:
                    # brown
                    img_gray[i][img_gray[i] <= 0.4] = 0
                    img_gray[i][img_gray[i] > 0.4] = 1
                else:
                    # grey
                    img_gray[i] = feature.canny(img_gray[i], sigma=5, low_threshold=0.2)

            edges = (2 * img_gray) - 1  # pointer magic
            return torch.from_numpy(edges[:, None, :, :])

        else:
            images = images.cpu().numpy()
            img_gray = np.squeeze(self.seg_sim(images))
            edges = np.array([feature.canny(img, sigma=3, low_threshold=0.2) for img in img_gray])
            edges = (2 * edges) - 1
            return torch.from_numpy(edges[:, None, :, :])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.data_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.data_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.y = input['B_class'].to(self.device)
        self.data_Seg = input['BSeg'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.output = self.netE(self.data_B)  # generate output image given the input data_A
        self.target = self.data_Seg#self.do_canny(self.data_B, False).cuda()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_MSE = self.MSE(self.output, self.target) * self.opt.lambda_regression
        self.loss_invSSIM = (1 - self.SSIM(self.output, self.target)) * self.opt.lambda_regression

        self.output = (self.output + 1) / 2
        self.target = (self.target + 1) / 2
        self.loss_BCE = self.BCE_loss(self.output, self.target) * self.opt.lambda_regression

        if self.edge_loss == 'BCE':
            self.loss_BCE.backward()
        elif self.edge_loss == 'MSE':
            self.loss_MSE.backward()  # calculate gradients of network G w.r.t. loss_G
        elif self.edge_loss == "SSIM":
            self.loss_invSSIM.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()  # first call forward to calculate intermediate results
        self.optimizer.zero_grad()  # clear network G's existing gradients
        self.backward()  # calculate gradients for network G
        self.optimizer.step()  # update gradients for network G
