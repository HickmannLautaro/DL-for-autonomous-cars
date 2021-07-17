import geomloss
import numpy as np
import torch
from kornia.enhance import histogram
from skimage import feature
from skimage.color import rgb2gray
from torch import nn
from torch.autograd import Variable
from torchvision import transforms

import util.util as util
from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

import matplotlib.pyplot as plt
from skimage.morphology import dilation, closing,white_tophat, disk

import pytorch_ssim


class ConditionalCUTModel(BaseModel):
    """ This class implements CUT and FastCUT model with different styles.
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--lambda_hist', type=float, default=0.0, help="weight for color histogram loss between generated and real image")
        parser.add_argument('--lambda_edge', type=float, default=0.0, help="weight for edge loss between generated image and simulated image")
        parser.add_argument('--edge_warmup', type=int, default=25, help="edge loss percentage increases from 0 to 1, i.e. edge_loss * (epoch /edge_warmup) for the first edge_warmup epochs")
        parser.add_argument('--edge_loss', type=str, default='MSE', choices=['MSE', 'BCE', 'SSIM'], help='what loss to use for edge detection comparison')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.epoch = 0
        self.edge_warmup = opt.edge_warmup
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        if self.opt.lambda_hist > 0:
            self.loss_names += ['H']
        if self.opt.lambda_edge > 0:
            self.edge_loss = opt.edge_loss
            self.loss_names += ['E']
            self.visual_names += ["real_E", "fake_E"]



        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        # self.netSeg = networks.define_Seg(self.gpu_ids)
        if self.opt.lambda_edge > 0.0:
            self.netE = networks.define_Seg(in_channels=opt.input_nc, out_channel=1, ngf=16,
                                            gpu_ids=self.gpu_ids)
            print("Using edge loss. Loading pre-trained edge detection network.")
            self.load_E()

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.ConditionalGANLoss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.lambda_edge > 0.0:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_E)

        self.binarize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            lambda x: x < 0,
            lambda x: x.int(),
        ])

    def update_epoch(self, epoch):
        self.epoch = epoch - 1  # -1 to turn off in first epoch

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.y = self.y[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()  # calculate gradients for D
            self.compute_G_loss().backward()  # calculate graidents for G
            # self.compute_C_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr,
                                                    betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def load_E(self):
        import os
        load_filename = '%s_net_%s.pth' % ("latest", "E")
        if self.opt.isTrain and self.opt.pretrained_name is not None:
            load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
        else:
            load_dir = self.save_dir

        load_path = os.path.join(load_dir, load_filename)
        net = getattr(self, 'net' + "E")
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)

        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)

        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        if self.opt.lambda_edge > 0:
            self.loss_E = self.opt.lambda_edge * self.compute_canny_loss()
            self.loss_G = self.loss_G + self.loss_E
        if self.opt.lambda_hist > 0:
            self.loss_H = self.opt.lambda_hist * self.compute_histogram_loss()
            self.loss_G = self.loss_G + self.loss_H

        self.loss_G.backward()
        self.optimizer_G.step()

        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        #        self.seg_A = input['Seg'].to(self.device)
        self.y = input['B_class'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.real = torch.cat((self.real_A, self.real_B),
                              dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A

        tmp_y = torch.cat((self.y, self.y), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.y

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        #     tmp_y = torch.cat((self.y,self.y))
        self.fake = self.netG((self.real, tmp_y))  # todo

        self.fake_B = self.fake[:self.real_A.size(0)]  # todo
        if self.opt.lambda_edge > 0:
            self.real_E = self.do_canny(self.real_A, True).cuda()

            self.fake_E = self.netE(self.fake_B)

        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]
        #  print(self.idt_B.shape)

    def seg_sim(self, image):

        aux2 = np.zeros((image.shape[0], 256, 256, 1), dtype=np.uint8)
        aux2[np.where((image[:, :256, :, 1] >= 98) & (image[:, :256, :, 1] <= 112))] = [255]
        aux2[np.where(image[:, :256, :, 1] == 107)] = [255]
        aux2[np.where(image[:, :256, :, 1] == 102)] = [255]
        aux2[np.where(image[:, :256, :, 1] == 111)] = [255]
        return aux2

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
            img = images.cpu().detach().numpy()
            img = ((np.transpose(img, [0, 2, 3, 1]) + 1) / 2 * 255).astype(np.uint8)

            # img_gray = rgb2gray(np.transpose(images, [0, 2, 3, 1]))
            img_gray = np.squeeze(self.seg_sim(img))

            # for i, _ in enumerate(img_gray):
            #     img_gray[i] = feature.canny(img_gray[i], sigma=5, low_threshold=0.2)

            edges = np.array([feature.canny(img, sigma=15, low_threshold=0.1) for img in img_gray]).astype(np.float32)
            #
            # fig, axes = plt.subplots(4, 3, figsize=(25, 6))
            # fig.suptitle("98 , 112")
            # for i, aux in enumerate(img):
            #     axes[i, 0].imshow(aux)
            #
            # for i, aux in enumerate(img_gray):
            #     axes[i, 1].imshow(aux)
            # for i, aux in enumerate(edges):
            #     axes[i, 2].imshow(aux)
            # plt.show()

            # edges = np.array([feature.canny(img, sigma=3, low_threshold=0.2) for img in img_gray])
            edges = (2 * edges) - 1  # pointer magic
            return torch.from_numpy(edges[:, None, :, :])

    def compute_histogram_loss(self):
        n_bins = 50
        bins = torch.torch.linspace(-1, 1, n_bins).cuda()
        B, C, W, H = self.real_B.shape

        # print(torch.max(self.real_B))
        self.real_H = torch.cat(
            [histogram(self.real_B[:, c, :, :].view(B, W * H), bins, bandwidth=torch.tensor(2 / n_bins).cuda()) for c in
             [0, 1, 2]], dim=1)
        self.fake_H = torch.cat(
            [histogram(self.fake_B[:, c, :, :].view(B, W * H), bins, bandwidth=torch.tensor(2 / n_bins).cuda()) for c in
             [0, 1, 2]], dim=1)
        #  from matplotlib import pyplot as plt
        # fig, axs = plt.subplots(2)
        # axs[0].plot(hist_real[0].cpu())
        # axs[1].plot(hist_fake[0].cpu().detach().numpy())
        # plt.show()
        criterion = geomloss.SamplesLoss("sinkhorn")
        # criterion = torch.nn.BCELoss()

        # loss = criterion(torch.flatten(self.fake_E)[:,None], torch.flatten(self.real_E)[:,None])
        loss = criterion(self.fake_H, self.real_H)
        loss = Variable(loss, requires_grad=True)
        return loss

    def compute_canny_loss(self):

        # criterion = SamplesLoss(loss="hausdorff")
        if self.edge_loss == 'BCE':
            self.real_E = (self.real_E + 1) / 2
            self.fake_E = (self.fake_E + 1) / 2
            criterion = torch.nn.BCELoss()
            loss = criterion(self.fake_E, self.real_E)
        elif self.edge_loss == 'MSE':
            criterion = torch.nn.MSELoss()

            loss = criterion(self.fake_E, self.real_E)
        elif self.edge_loss == "SSIM":

            criterion = pytorch_ssim.SSIM(window_size=11)
            loss = 1 - criterion(self.fake_E, self.real_E)

        # loss = criterion(torch.flatten(self.fake_E)[:,None], torch.flatten(self.real_E)[:,None])
        if self.edge_warmup > 0:
            if self.epoch <= self.edge_warmup:
                loss = loss * (self.epoch / self.edge_warmup)

        loss = Variable(loss, requires_grad=True)
        return loss

    def compute_seg_loss(self, fake=False):
        if fake:
            #     for param in self.netC.parameters(): # dont update classifier on fake data; still propagate through
            #        param.requires_grad = False

            c_pred = self.netSeg(self.fake_B)
            self.sB = torch.argmax(torch.nn.LogSoftmax(dim=1)(c_pred), dim=1, keepdim=True)



        else:
            #     for param in self.netC.parameters(): # update classifier on real data
            #       param.requires_grad = True
            c_pred = self.netSeg(self.real_A)
            # c_pred = torch.argmax(nn.Sigmoid()(c_pred), dim=1, keepdim=True)
            self.sA = torch.argmax(torch.nn.LogSoftmax(dim=1)(c_pred), dim=1, keepdim=True)

        from torch.autograd import Variable
        # loss = nn.BCELoss()
        criterion = nn.CrossEntropyLoss(reduction="mean")
        target = ((self.seg_A[:, 0] + 1) / 2).type(torch.LongTensor).cuda()
        # images = self.real_A.cpu()
        # images = torch.stack([self.binarize(image) for image in images]).to(self.y.device)
        loss = criterion(c_pred, target)
        #   loss = loss(c_pred.float(), images.float())
        loss = Variable(loss, requires_grad=True)
        return loss
        """def compute_C_loss(self, fake=False):

        if fake:
            #     for param in self.netC.parameters(): # dont update classifier on fake data; still propagate through
            #        param.requires_grad = False

            c_pred = self.netC(self.fake_B)

        else:
            #     for param in self.netC.parameters(): # update classifier on real data
            #       param.requires_grad = True
            c_pred = self.netC(self.real_B)

        c_real = torch.argmax(self.y, dim=1)

        from torch.autograd import Variable
        loss = nn.CrossEntropyLoss()

        loss = loss(c_pred, c_real)
        loss = Variable(loss, requires_grad=True)
        return loss
    """

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)

        self.loss_D_fake = self.criterionGAN(pred_fake, None).mean()
        # Real
        self.pred_real = self.netD(self.real_B)

        loss_D_real = self.criterionGAN(self.pred_real, self.y)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, self.y).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:

            self.loss_NCE = self.calculate_NCE_loss((self.real_A, self.y), (self.fake_B, self.y))
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:

            self.loss_NCE_Y = self.calculate_NCE_loss((self.real_B, self.y), (self.idt_B, self.y))
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):

        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
