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
import numpy as np
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from util.util import prepare_z_y, one_hot, visualize_imgs
from torch.distributions import Categorical
from collections import OrderedDict

# from TTUR import fid
# from util.inception import get_inception_score
# from inception_pytorch import inception_utils

import copy
import math

class CDEGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        #parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset
        if is_train:
            parser.add_argument('--g_loss_mode', nargs='*', default=['nsgan','lsgan','vanilla'], help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--d_loss_mode', nargs='*', default=['vanilla'], help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)')

            parser.add_argument('--lambda_f', type=float, default=0.1, help='the hyperparameter that balance Fq and Fd')
            parser.add_argument('--g_candi_num', type=int, default=1, help='# of survived generator candidatures in each evolutinary iteration.')
            parser.add_argument('--d_candi_num', type=int, default=2, help='# of survived discriminator candidatures in each evolutinary iteration.')
            parser.add_argument('--eval_size', type=int, default=64, help='batch size during each evaluation.')

            parser.add_argument('--lambda_weight', type=float, default=1.0, help='batch size during each evaluation.')
            parser.add_argument('--mean_type', type=str, default='arithmetic', help='arithmetic | geometric | harmonic')
            parser.add_argument('--weight_type', type=str, default='normal', help='normal | log')

            parser.add_argument('--D_different', default=True, help='if different discriminators sample are different for D update')
            parser.add_argument('--G_different', default=True, help='if different discriminators sample are different for G update')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.loss_names = ['D_real', 'D_fake', 'D_gp', 'D', 'G']
        self.visual_names = ['real_visual', 'gen_visual']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # define networks

        self.print_mutaion = []
        if self.isTrain:  # only defined during training time
            # define G mutations
            self.G_mutations = []
            for g_loss in opt.g_loss_mode:
                self.G_mutations.append(networks.GANLoss(g_loss, 'G', opt.which_D).to(self.device))
            # define D mutations
            self.D_mutations = []
            for d_loss in opt.d_loss_mode:
                self.D_mutations.append(networks.GANLoss(d_loss, 'D', opt.which_D).to(self.device))

            # vanilla discriminator criterion for calculating fitness score
            self.vanilla_Dcriterion = networks.GANLoss('vanilla', 'D', opt.which_D).to(self.device)


        self.netG = networks.define_G(opt.z_dim, opt.output_nc, opt.ngf, opt.netG,
                                        opt.g_norm, opt.cgan, opt.cat_num, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
        self.optimizers.append(self.optimizer_G)
        self.G_candis = []
        self.optG_candis = []
        for i in range(opt.g_candi_num):
            self.G_candis.append(copy.deepcopy(self.netG.state_dict()))
            self.optG_candis.append(copy.deepcopy(self.optimizer_G.state_dict()))


        self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.d_norm, opt.cgan, opt.cat_num, opt.init_type, opt.init_gain, self.gpu_ids)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
        self.optimizers.append(self.optimizer_D)
        self.D_candis = []
        self.optD_candis = []
        for i in range(opt.d_candi_num):
            netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.d_norm, opt.cgan, opt.cat_num, opt.init_type, opt.init_gain, self.gpu_ids)
            optimD = torch.optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
            self.D_candis.append(netD)
            self.optD_candis.append(optimD)

        # visulize settings
        self.N =int(np.trunc(np.sqrt(min(opt.batch_size, 64))))
        if self.opt.z_type == 'Gaussian':
            self.z_fixed = torch.randn(self.N*self.N, opt.z_dim, 1, 1, device=self.device)
        elif self.opt.z_type == 'Uniform':
            self.z_fixed = torch.rand(self.N*self.N, opt.z_dim, 1, 1, device=self.device)*2. - 1.

        # the # of image for each evluation
        # TODO batch size details
        self.eval_size = max(math.ceil((opt.batch_size * opt.D_iters) / opt.d_candi_num), opt.eval_size)

    def set_input(self, input):
        self.input_imgs = input['image'].to(self.device)
        # self.real_imgs just for visualizing
        self.real_imgs = self.input_imgs[:self.opt.batch_size]

    # TODO
    def optimize_parameters(self, iters):
        self.g_useless_real_score = self.netD(self.input_imgs[:self.opt.batch_size])
        for i in range(self.opt.D_iters + 1):
            if i == 0:
                self.G_Fitness, self.evalimgs, self.g_sel_mut = self.Evo_G()
                self.print_mutaion.append(self.g_sel_mut)
            else:
                real_imgs = self.input_imgs[(i-1)*self.opt.batch_size*self.opt.d_candi_num: i*self.opt.batch_size*self.opt.d_candi_num]
                fake_imgs = self.evalimgs[(i-1)*self.opt.batch_size*self.opt.d_candi_num: i*self.opt.batch_size*self.opt.d_candi_num]
                self.D_Fitness, self.d_sel_mut = self.Evo_D(fake_imgs, real_imgs)
        if iters %  self.opt.print_freq == 0 and iters > 0:
            print('iters:', iters, 'g_sel_mul:', self.g_sel_mut, 'd_sel_mul:', self.d_sel_mut)



    def Evo_D(self, fake_imgs, real_imgs):
        eval_real_imgs = self.input_imgs[-self.opt.eval_size:]
        eval_fake_imgs = self.evalimgs[-self.opt.eval_size:]

        F_list = -100*np.ones(self.opt.d_candi_num)
        D_F = []
        D_list = []
        optD_list = []
        selected_mutation = []
        count = 0
        for i in range(self.opt.d_candi_num):
            if self.opt.D_different:
                train_fake_imgs = fake_imgs[i*self.opt.batch_size:(i+1)*self.opt.batch_size]
                train_real_imgs = real_imgs[i*self.opt.batch_size:(i+1)*self.opt.batch_size]
            else:
                train_fake_imgs = real_imgs[:self.opt.batch_size]
                train_real_imgs = real_imgs[:self.opt.batch_size]

            for j, criterionD in enumerate(self.D_mutations):
                # backward and update
                self.netD.load_state_dict(self.D_candis[i].state_dict())
                self.optimizer_D.load_state_dict(self.optD_candis[i].state_dict())
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()
                self.loss_D, self.loss_D_fake, self.loss_D_real, self.loss_D_gp = self.backward_D(
                    self.netD, train_fake_imgs, train_real_imgs, criterionD)
                self.optimizer_D.step()

                # Evaluation
                Fd_D = self.D_diverse_score(self.netD, eval_fake_imgs, eval_real_imgs)
                # selection
                if count < self.opt.d_candi_num:
                    F_list[count] = Fd_D
                    D_F.append(Fd_D)
                    D_list.append(copy.deepcopy(self.netD))
                    optD_list.append(copy.deepcopy(self.optimizer_D))
                    selected_mutation.append(self.opt.d_loss_mode[j])
                else:
                    fit_com = Fd_D - F_list
                    if min(fit_com) < 0:
                        ids_replace = np.where(fit_com==max(fit_com))[0][0]
                        F_list[ids_replace] = Fd_D
                        D_F[ids_replace] = Fd_D
                        D_list[ids_replace] = copy.deepcopy(self.netD)
                        optD_list[ids_replace] = copy.deepcopy(self.optimizer_D)
                        selected_mutation[ids_replace] = self.opt.d_loss_mode[j]
                count +=1
        self.D_candis = copy.deepcopy(D_list)
        self.optD_candis = copy.deepcopy(optD_list)
        return np.array(D_F), selected_mutation


    def D_diverse_score(self, netD, fake_imgs, real_imgs):
        self.set_requires_grad(netD, True)
        fake_score = netD(fake_imgs)
        real_score = netD(real_imgs)

        # Diversity fitness score
        fake_loss, real_loss = self.vanilla_Dcriterion(fake_score, real_score)
        eval_loss_D = fake_loss + real_loss
        gradients = torch.autograd.grad(outputs=eval_loss_D, inputs=netD.parameters(),
                                        grad_outputs=torch.ones(eval_loss_D.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad,grad])
        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        return Fd

    def backward_D(self, netD, fake_imgs, real_imgs, criterionD):
        fake_score = netD(fake_imgs)
        real_score = netD(real_imgs)
        loss_D_fake, loss_D_real = criterionD(fake_score, real_score)

        # if self.opt.use_gp is True:
        if 1 == 1:
            loss_D_gp = networks.cal_gradient_penalty(netD, real_imgs, fake_imgs, self.device, type='mixed', constant=1.0, lambda_gp=10.0)[0]
        else:
            loss_D_gp = 0.
        loss_D = loss_D_fake + loss_D_real + loss_D_gp
        loss_D.backward()
        return loss_D, loss_D_fake, loss_D_real, loss_D_gp

    def Evo_G(self):
        eval_real_imgs = self.input_imgs[-self.eval_size:,:,:,:]
        F_list = np.zeros(self.opt.g_candi_num)
        Fit_list = []
        G_list = []
        optG_list = []
        evalimg_list = []
        selected_mutation = []
        count = 0
        # variation-evluation-selection
        for i in range(self.opt.g_candi_num):
            for j, criterionG in enumerate(self.G_mutations):
                # Variation
                self.netG.load_state_dict(self.G_candis[i])
                self.optimizer_G.load_state_dict(self.optG_candis[i])
                self.optimizer_G.zero_grad()
                self.loss_G = self.boosted_backward_G(criterionG)
                self.optimizer_G.step()
                # Evaluation
                with torch.no_grad():
                    eval_fake_imgs = self.forward(batch_size=self.eval_size+self.opt.D_iters*self.opt.batch_size*self.opt.d_candi_num).detach()
                Fq, Fd = self.fitness_score(eval_fake_imgs[:self.eval_size], eval_real_imgs)
                F = Fq + self.opt.lambda_f * Fd
                # Selection
                if count < self.opt.g_candi_num:
                    F_list[count] = F
                    Fit_list.append([Fq, Fd, F])
                    G_list.append(copy.deepcopy(self.netG.state_dict()))
                    optG_list.append(copy.deepcopy(self.optimizer_G.state_dict()))
                    evalimg_list.append(eval_fake_imgs)
                    selected_mutation.append(self.opt.g_loss_mode[j])
                else:
                    fit_com = F - F_list
                    if max(fit_com) > 0:
                        ids_replace = np.where(fit_com==max(fit_com))[0][0]
                        F_list[ids_replace] = F
                        Fit_list[ids_replace] = [Fq, Fd, F]
                        G_list[ids_replace] = copy.deepcopy(self.netG.state_dict())
                        optG_list[ids_replace] = copy.deepcopy(self.optimizer_G.state_dict())
                        evalimg_list[ids_replace] = eval_fake_imgs
                        selected_mutation[ids_replace] = self.opt.g_loss_mode[j]
                count += 1
        self.G_candis = copy.deepcopy(G_list)
        self.optG_candis = copy.deepcopy(optG_list)

        # shuffle
        evalimg_list = torch.cat(evalimg_list, dim=0)
        shuffle_ids = torch.randperm(evalimg_list.size()[0])
        evalimg_list = evalimg_list[shuffle_ids]
        return np.array(Fit_list), evalimg_list, selected_mutation

    def boosted_backward_G(self, criterionG):
        gen_imgs = self.forward(batch_size=self.opt.d_candi_num*self.opt.batch_size)
        g_losses = []
        for i in range(self.opt.d_candi_num):
            netD = self.D_candis[i]
            self.set_requires_grad(netD, False)

            if self.opt.G_different:
                fake_logits = netD(gen_imgs[i*self.opt.batch_size:(i+1)*self.opt.batch_size])
            else:
                fake_logits = netD(gen_imgs[:self.opt.batch_size])

            loss_G_fake, loss_G_real = criterionG(fake_logits, self.g_useless_real_score)
            loss_G = loss_G_fake + loss_G_real
            g_losses.append(loss_G)
        loss_G = self.mix_prediction(g_losses)
        loss_G.backward()
        return loss_G


    def mix_prediction(self, losses):
        lam = self.opt.lambda_weight
        assert self.opt.mean_type in ['arithmetic','geometric','harmonic', 'sum']
        assert self.opt.weight_type in ['normal','log']
        if lam == 0.:
            weights = torch.ones_like(losses)
        else:
            if self.opt.weight_type == 'log':
                weights = [torch.pow(losses[i], lam).detach() for i in range(len(losses))]
            else:
                weights = [torch.exp(lam * losses[i]).detach() for i in range(len(losses))]
        weights_normed = [weights[i]/sum(weights) for i in range(len(weights))]
        loss = 0
        if self.opt.mean_type == 'arithmetic':
            loss = np.sum([weights_normed[l]*losses[l] for l in losses])
        elif self.opt.mean_type == 'sum':
            loss = np.sum(losses)
        else:
            raise NotImplementedError('other mean_type in mix_prediction not implemented')
        return loss



    def forward(self, batch_size = None):
        bs = self.opt.batch_size if batch_size is None else batch_size
        if self.opt.z_type == 'Gaussian':
            z = torch.randn(bs, self.opt.z_dim, 1, 1, device=self.device)
        elif self.opt.z_type == 'Uniform':
            z = torch.rand(bs, self.opt.z_dim, 1, 1, device=self.device)*2. - 1.
        else:
            raise NotImplementedError('z type not implemented')
        # Fake images
        gen_imgs = self.netG(z)
        return gen_imgs

    def fitness_score(self, eval_fake_imgs, eval_real_imgs):
        eval_fake = []
        # eval_real = []
        Fd = 0
        for i in range(self.opt.d_candi_num):
            netD = self.D_candis[i]
            self.set_requires_grad(netD, True)
            eval_fake_score = netD(eval_fake_imgs)
            eval_real_score = netD(eval_real_imgs)
            eval_fake.extend(eval_fake_score)
            Fd += self.calculate_gradient(netD, eval_fake_score, eval_real_score)
        eval_fake = torch.tensor(eval_fake)
        Fq = nn.Sigmoid()(eval_fake).data.mean().cpu().numpy()
        return Fq, Fd

    def calculate_gradient(self, netD, fake_score, real_score):
        eval_D_fake, eval_D_real = self.vanilla_Dcriterion(fake_score, real_score)
        eval_D = eval_D_fake + eval_D_real
        gradients = torch.autograd.grad(outputs=eval_D, inputs=netD.parameters(),
                                        grad_outputs=torch.ones(eval_D.size()).to(self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        with torch.no_grad():
            for i, grad in enumerate(gradients):
                grad = grad.view(-1)
                allgrad = grad if i == 0 else torch.cat([allgrad,grad])
        Fd = -torch.log(torch.norm(allgrad)).data.cpu().numpy()
        return Fd
