"""
Pix2pixHD model with additional image data as input
It basically takes additional inputs of cropped image, which is then used as
additional 3-channel input.
The additional image input has "hole" in the regions inside object bounding box,
which is subsequently filled by the generator network.
"""
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from .layer_util import *
import util.util as util
from collections import OrderedDict
from seg.data_process import id2label_tensor, label2id_tensor
from seg.segment import DRNSeg
import math

NULLVAL = 0.0

# TODO(sh): change the forward_wrapper things to enable multi-gpu training
# TODO(sh): add additional input of "mask_in" for the binary mask of object region
# TODO(sh): enlarge the context marign

class houdini_loss(nn.Module):
    def __init__(self, use_cuda=True, num_class=19, ignore_index=None):
        super(houdini_loss, self).__init__()
        # self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        self.use_cuda = use_cuda
        self.num_class = num_class
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        pred = logits.max(1)[1].data
        target = target.data
        size = list(target.size())
        if self.ignore_index is not None:
            pred[pred == self.ignore_index] = self.num_class
            target[target == self.ignore_index] = self.num_class
        pred = torch.unsqueeze(pred, dim=1)
        target = torch.unsqueeze(target, dim=1)
        size.insert(1, self.num_class+1)
        pred_onehot = torch.zeros(size)
        target_onehot = torch.zeros(size)
        if self.use_cuda:
            pred_onehot = pred_onehot.cuda()
            target_onehot = target_onehot.cuda()
        pred_onehot = pred_onehot.scatter_(1, pred, 1).narrow(1, 0, self.num_class)
        target_onehot = target_onehot.scatter_(1, target, 1).narrow(1, 0, self.num_class)
        pred_onehot = Variable(pred_onehot)
        target_onehot = Variable(target_onehot)
        neg_log_softmax = -F.log_softmax(logits, dim=1)
        # print(logits.size())
        # print(neg_log_softmax.size())
        # print(target_onehot.size())
        twod_cross_entropy = torch.sum(neg_log_softmax*target_onehot, dim=1)
        pred_score = torch.sum(logits*pred_onehot, dim=1)
        target_score = torch.sum(logits*target_onehot, dim=1)
        mask = 0.5 + 0.5 * (((pred_score-target_score)/math.sqrt(2)).erf())
        return torch.mean(mask * twod_cross_entropy)

class Pix2PixHDModel_condImgDoubleAdv(BaseModel):
    def __init__(self, opt):
        super(Pix2PixHDModel_condImgDoubleAdv, self).__init__(opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.netG_type = opt.netG
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        # NOTE(sh): 3-channels for adddional rgb-image
        input_nc = opt.label_nc if opt.label_nc != 0 else 3 

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc         
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        from .Pix2Pix_NET import GlobalGenerator, GlobalTwoStreamGenerator
        if opt.netG=='global':
            netG_input_nc += 3
            self.netG = GlobalGenerator(netG_input_nc, opt.output_nc, opt.ngf, 
                    opt.n_downsample_global, opt.n_blocks_global, opt.norm, 'reflect', opt.use_output_gate)
        elif opt.netG=='global_twostream':
            self.netG = GlobalTwoStreamGenerator(netG_input_nc, opt.output_nc, opt.ngf, 
                    opt.n_downsample_global, opt.n_blocks_global, opt.norm, 'reflect', opt.use_skip, 
                    opt.which_encoder, opt.use_output_gate, opt.feat_fusion)
        else:
            raise NameError('global generator name is not defined properly: %s' % opt.netG)
        print(self.netG)
        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())   
            self.netG.cuda(opt.gpu_ids[0])
        self.netG.apply(weights_init)

        # Discriminator network
        if self.isTrain:
            self.no_imgCond = opt.no_imgCond
            self.mask_gan_input = opt.mask_gan_input
            self.use_soft_mask = opt.use_soft_mask
            use_sigmoid = opt.no_lsgan
            if self.no_imgCond:
                netD_input_nc = input_nc + opt.output_nc 
            else:
                netD_input_nc = input_nc + 3 + opt.output_nc 
            if not opt.no_instance:
                netD_input_nc += 1
            if opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
                netD_input_nc = 3
            from .Discriminator_NET import MultiscaleDiscriminator
            self.netD = MultiscaleDiscriminator(netD_input_nc, opt.ndf, opt.n_layers_D, 
                    opt.norm, use_sigmoid, opt.num_D, not opt.no_ganFeat_loss)
            print(self.netD)
            if len(opt.gpu_ids) > 0:
                assert(torch.cuda.is_available())   
                self.netD.cuda(opt.gpu_ids[0])
            self.netD.apply(weights_init)

        ### Encoder network
        if self.gen_features:          
            from .Pix2Pix_NET import Encoder
            self.netE = Encoder(opt.output_nc, opt.feat_num, opt.nef, opt.n_downsample_E, opt.norm)
            print(self.netE)
            if len(opt.gpu_ids) > 0:
                assert(torch.cuda.is_available())   
                self.netE.cuda(opt.gpu_ids[0])
            self.netE.apply(weights_init)

        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            from .losses import GANLoss, VGGLoss
            self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]                            
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        # init segnet
        seg_mean = torch.FloatTensor([0.29010095242892997, 0.32808144844279574, 0.28696394422942517]).view(1, 3, 1, 1)
        self.seg_mean = Variable(seg_mean.cuda())
        seg_std = torch.FloatTensor([0.1829540508368939, 0.18656561047509476, 0.18447508988480435]).view(1, 3, 1, 1)
        self.seg_std = Variable(seg_std.cuda())
        single_model = DRNSeg('drn_d_22', 19, pretrained_model=None, pretrained=False)
        # single_model.load_state_dict(torch.load('pretrain/drn_d_22_cityscapes.pth'))
        # self.netS = nn.DataParallel(single_model).cuda()
        self.netS = torch.nn.DataParallel(single_model)
        # self.netS.load_state_dict(torch.load('./pretrain/model_best.pth.tar')['state_dict'])
        self.netS.load_state_dict(torch.load('./pretrain//checkpoint_200.pth.tar')['state_dict'])

        self.netS = self.netS.cuda()
        # init attack
        self.houdini_loss = houdini_loss(ignore_index=255)

    def name(self):
        return 'Pix2PixHDModel_condImgDoubleAdv'

    def encode_input(self, label_map, label_map1, inst_map=None, inst_map1=None, real_image=None, feat_map=None, mask_in=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
            input_label1 = label_map1.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            # [1, 1, 256, 256] (1, 28)
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_() # [1, 35, 256, 256] (0, 1)
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            input_label1 = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label1 = input_label1.scatter_(1, label_map1.data.long().cuda(), 1.0)


        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            inst_map1 = inst_map1.data.cuda()
            edge_map1 = self.get_edges(inst_map1)
            input_label = torch.cat((input_label, edge_map), dim=1)
            input_label1 = torch.cat((input_label1, edge_map1), dim=1)
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        assert(real_image is not None)
        assert(mask_in is not None)
        real_image = Variable(real_image.data.cuda())
        mask_object_box = mask_in.repeat(1,3,1,1).cuda()
        cond_image = (1 - mask_object_box) * real_image + mask_object_box * NULLVAL # TODO(sh): define null_img 
        
        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, input_label1, inst_map, inst_map1, real_image, feat_map, cond_image

    def discriminate(self, input_label, test_image, mask, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if self.opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
            input_concat = test_image.detach()
        if self.mask_gan_input:
            input_concat = input_concat * mask.repeat(1, input_concat.size(1), 1, 1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward_wrapper(self, data, infer=False):
        label = Variable(data['label'])
        inst = Variable(data['inst'])
        image = Variable(data['image'])
        mask_in = Variable(data['mask_in'])
        mask_out = Variable(data['mask_out'])
        feat = None
        losses, generated = self.forward(label, inst, image, feat, mask_in, mask_out, infer)
        return losses, generated 
        
    def forward(self, label, label1, inst, inst1, image, feat, mask_in, mask_out, infer=False):
        # Encode Inputs
        input_label, input_label1, inst_map, inst_map1, real_image, feat_map, cond_image = self.encode_input(label, label1, inst, inst1, image, feat, mask_in=mask_in)

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_label = torch.cat((input_label, cond_image), 1)
        # Fake Generation
        input_concat = input_label  
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
        elif self.netG_type == 'global_twostream':
            fake_image = self.netG.forward(cond_image, input_mask, mask_in)

        # Fake Detection and Loss
        if self.no_imgCond:
            netD_cond = input_mask
        else:
            netD_cond = input_label
        mask_cond = mask_in if not self.use_soft_mask else mask_out
        pred_fake_pool = self.discriminate(netD_cond, fake_image, mask_cond, True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss        
        pred_real = self.discriminate(netD_cond, real_image, mask_cond, False)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        netD_in = torch.cat((netD_cond, fake_image), dim=1)
        if self.opt.netG=='global_twostream' and self.opt.which_encoder=='ctx':
            netD_in = fake_image
        if self.mask_gan_input:
            netD_in = netD_in * mask_cond.repeat(1, netD_in.size(1), 1, 1)
        pred_fake = self.netD.forward(netD_in)        
        loss_G_GAN = self.criterionGAN(pred_fake, True)         
        
        # GAN feature matching loss
        loss_G_GAN_Feat = Variable(self.Tensor([0]))
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = Variable(self.Tensor([0]))
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        # color matching loss
        if self.opt.lambda_rec > 0:
            # TOOD(sh): this part is bit hacky but let's leave it for now 
            loss_G_GAN_Feat += self.criterionFeat(fake_image, real_image.detach()) * self.opt.lambda_rec
        
        self.fake_image = fake_image.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_image = cond_image.cpu().data[0] 

        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ], None if not infer else fake_image ]

    def inference(self, label, label1, inst, inst1, image, mask_in, mask_out):
        # Encode Inputs        
        input_label, input_label1, inst_map, inst_map1, real_image, _, cond_image = self.encode_input(label, label1, inst, inst1, image, mask_in=mask_in, infer=True)
        mask_in = mask_in.cuda()

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_mask1 = input_label1.clone()
        input_label = torch.cat((input_label, cond_image), 1)
        input_label1 = torch.cat((input_label1, cond_image), 1)
        
        # Fake Generation
        input_concat = input_label  
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
            fake_image1 = self.netG.forward(input_label1, mask_in)
        elif self.netG_type == 'global_twostream':
            mask_in = mask_in.cuda()
            fake_image = self.netG.forward(cond_image, input_mask, mask_in)
            fake_image1 = self.netG.forward(cond_image, input_mask1, mask_in)

        self.fake_image = fake_image.cpu().data[0]
        self.fake_image1 = fake_image1.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_label1 = input_mask1.cpu().data[0]
        self.input_image = cond_image.cpu().data[0] 

        return fake_image

    def interp(self, label, label1, inst, inst1, image, mask_in, mask_out):
        # Encode Inputs
        input_label, input_label1, inst_map, inst_map1, real_image, _, cond_image = self.encode_input(label, label1,
                                                                                                      inst, inst1,
                                                                                                      image,
                                                                                                      mask_in=mask_in,
                                                                                                      infer=True)
        mask_in = mask_in.cuda()

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_mask1 = input_label1.clone()
        input_label = torch.cat((input_label, cond_image), 1)
        input_label1 = torch.cat((input_label1, cond_image), 1)

        # Fake Generation
        input_concat = input_label
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
            fake_image1 = self.netG.forward(input_label1, mask_in)
        elif self.netG_type == 'global_twostream':
            mask_in = mask_in.cuda()
            fake_feature, ctx_feats = self.netG.g_in(cond_image, input_mask, mask_in)
            fake_feature1, ctx_feats1 = self.netG.g_in(cond_image, input_mask1, mask_in)
            fake_image = self.netG.forward(cond_image, input_mask1, mask_in)
            fake_image1 = self.netG.g_out((fake_feature*0.2 + fake_feature1*0.8), ctx_feats, cond_image, mask_in)


        self.fake_image = fake_image.cpu().data[0]
        self.fake_image1 = fake_image1.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_label1 = input_mask1.cpu().data[0]
        self.input_image = cond_image.cpu().data[0]

        return fake_image

    def interp_attack(self, label, label1, label2, label3, inst, inst1, inst2, image, mask_in, mask_in2, mask_out, mask_target):
        # Encode Inputs
        original_labels = id2label_tensor(label).long().cuda()
        target_labels = id2label_tensor(label3).long().cuda()

        input_label, input_label1, inst_map, inst_map1, real_image, _, cond_image = self.encode_input(label, label1,
                                                                                                      inst, inst1,
                                                                                                      image,
                                                                                                      mask_in=mask_in,
                                                                                                      infer=True)
        _input_label, input_label2, _inst_map, inst_map2, _real_image, __, cond_image2 = self.encode_input(label, label2,
                                                                                                      inst, inst2,
                                                                                                      image,
                                                                                                      mask_in=mask_in2,
                                                                                                      infer=True)

        mask_in = mask_in.cuda()
        mask_in2 = mask_in2.cuda()
        mask_target = mask_target.cuda()

        # NOTE(sh): modified with additional image input
        input_mask = input_label.clone()
        input_mask1 = input_label1.clone()
        input_mask2 = input_label2.clone()
        input_label = torch.cat((input_label, cond_image), 1)
        input_label1 = torch.cat((input_label1, cond_image), 1)
        input_label2 = torch.cat((input_label2, cond_image2), 1)

        # Fake Generation
        input_concat = input_label
        if self.netG_type == 'global':
            fake_image = self.netG.forward(input_concat, mask_in)
            fake_image1 = self.netG.forward(input_label1, mask_in)
            fake_image2 = self.netG.forward(input_label2, mask_in)
        elif self.netG_type == 'global_twostream':
            fake_feature, ctx_feats = self.netG.g_in(cond_image, input_mask, mask_in)
            fake_feature1, ctx_feats1 = self.netG.g_in(cond_image, input_mask1, mask_in)
            fake_feature2, ctx_feats2 = self.netG.g_in(cond_image2, input_mask2, mask_in2)
            fake_image = self.netG.forward(cond_image, input_mask1, mask_in)
            fake_image1 = self.netG.g_out((fake_feature*0.2 + fake_feature1*0.8), ctx_feats, cond_image, mask_in)
            fake_image2 = self.netG.g_out((fake_feature * 0.2 + fake_feature2 * 0.8), ctx_feats, cond_image2, mask_in2)

        normed_fake_image = ((fake_image1 + 1.0)/2 -self.seg_mean)/self.seg_std
        logits = self.netS(real_image)[0]
        # logits = self.netS(normed_fake_image)[0]
        init_pred = torch.max(logits, 1)[1]
        print('ori_acc: %.3f' % ((init_pred == original_labels).cpu().data.numpy().sum() / (256 * 256)))


        # # pixel attack starts
        # ori_image = (real_image.clone()+ 1.0)/2
        # noise = torch.zeros(real_image.size()).cuda()
        # noise = Variable(noise, requires_grad=True)
        # noise_optimizer = torch.optim.Adam([noise], lr=1e-2)
        # mask_logits = mask_target.repeat(1, 19, 1, 1)
        # for i in range(20):
        #     noise_optimizer.zero_grad()
        #     self.netS.zero_grad()
        #     self.houdini_loss.zero_grad()
        #
        #     # x_hat = torch.clamp(ori_image + noise, 0.0, 1.0)
        #     x_hat = torch.clamp(ori_image + noise * mask_in, 0.0, 1.0)
        #     x_normal = (x_hat - self.seg_mean) / self.seg_std
        #     logits = self.netS(x_normal)[0]
        #     # hou_loss = self.houdini_loss(logits,target_labels.squeeze(1)) * 10
        #     hou_loss = self.houdini_loss(logits * mask_logits, target_labels.squeeze(1) * mask_target.squeeze(1).long()) * 10
        #     pred = torch.max(logits, 1)[1]
        #     print('acc: %.3f' % ((pred == target_labels).cpu().data.numpy().sum() / (256 * 256)))
        #     print('iteration %d loss %.3f' % (int(i), hou_loss.cpu().data.numpy()))
        #     hou_loss.backward()
        #     noise_optimizer.step()
        # # pixel attack ends



        # semantic attack starts
        alpha = torch.zeros(fake_feature.size()).cuda() + 0.8
        alpha = Variable(alpha, requires_grad=True)
        alpha2 = torch.zeros(fake_feature.size()).cuda() + 0.8
        alpha2 = Variable(alpha2, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([alpha,alpha2], lr=0.01)
        fake_feature_const = fake_feature.detach().clone()
        fake_feature1_const = fake_feature1.detach().clone()
        fake_feature2_const = fake_feature2.detach().clone()
        # ctx_feats_const = ctx_feats.detach().clone()
        mask_logits = mask_target.repeat(1, 19, 1, 1)
        for i in range(20):
            alpha_optimizer.zero_grad()
            self.netS.zero_grad()
            self.houdini_loss.zero_grad()

            # x_hat = torch.clamp(ori_image + noise, 0.0, 1.0)
            alpha_in = torch.clamp(alpha, 0.6, 1.0)
            alpha_in2 = torch.clamp(alpha2, 0.6, 1.0)
            semantic_image = self.netG.g_double_out((fake_feature_const * (1-alpha_in) + fake_feature1_const * alpha_in), (fake_feature_const * (1-alpha_in2) + fake_feature2_const * alpha_in2), ctx_feats, cond_image, mask_in, mask_in2)
            x_hat = (semantic_image + 1.0) / 2
            x_normal = (x_hat - self.seg_mean) / self.seg_std
            logits = self.netS(x_normal)[0]
            # hou_loss = self.houdini_loss(logits,target_labels.squeeze(1)) * 10
            hou_loss = self.houdini_loss(logits * mask_logits, target_labels.squeeze(1) * mask_target.squeeze(1).long()) * 10
            pred = torch.max(logits, 1)[1]

            if i ==0:
                ori_predict_map = label2id_tensor(pred.unsqueeze(1))
                ori_size = ori_predict_map.size()
                ori_oneHot_size = (ori_size[0], self.opt.label_nc, ori_size[2], ori_size[3])
                ori_predict_label = torch.cuda.FloatTensor(torch.Size(ori_oneHot_size)).zero_()
                self.ori_predict_label = ori_predict_label.scatter_(1, ori_predict_map.data.long().cuda(), 1.0).cpu().data[0]

            print('acc: %.3f' % ((pred == target_labels).cpu().data.numpy().sum() / (256 * 256)))
            print('iteration %d loss %.3f' % (int(i), hou_loss.cpu().data.numpy()))
            hou_loss.backward()
            alpha_optimizer.step()
        # semantic attack ends

        init_predict_map = label2id_tensor(init_pred.unsqueeze(1))
        predict_map = label2id_tensor(pred.unsqueeze(1))
        target_map = label2id_tensor(target_labels)
        size = predict_map.size()
        oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
        predict_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        self.predict_label = predict_label.scatter_(1, predict_map.data.long().cuda(), 1.0).cpu().data[0]
        init_predict_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        self.init_predict_label = init_predict_label.scatter_(1, init_predict_map.data.long().cuda(), 1.0).cpu().data[0]
        target_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        self.target_label = target_label.scatter_(1, target_map.data.long().cuda(), 1.0).cpu().data[0]

        self.fake_image = fake_image.cpu().data[0]
        self.fake_image1 = fake_image1.cpu().data[0]
        self.fake_image2 = fake_image2.cpu().data[0]
        self.real_image = real_image.cpu().data[0]
        self.input_label = input_mask.cpu().data[0]
        self.input_label1 = input_mask1.cpu().data[0]
        self.input_label2 = input_mask2.cpu().data[0]
        self.input_image = cond_image.cpu().data[0]
        self.input_image2 = cond_image2.cpu().data[0]
        self.perturb_image = ((x_hat-0.5)*2).cpu().data[0]

        return fake_image


    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def get_current_visuals(self):
        return OrderedDict([
            ('input_image', util.tensor2im(self.input_image)),
            ('input_image2', util.tensor2im(self.input_image2)),
            ('real_image', util.tensor2im(self.real_image)),
            ('synthesized_image', util.tensor2im(self.fake_image)),
            ('perturb_image', util.tensor2im(self.perturb_image)),
            ('synthesized_image1', util.tensor2im(self.fake_image1)),
            ('synthesized_image2', util.tensor2im(self.fake_image2)),
            ])

    def get_current_visuals1(self):
        return OrderedDict([
            ('input_label', util.tensor2label(self.input_label, self.opt.label_nc)),
            ('init_predict_label', util.tensor2label(self.init_predict_label, self.opt.label_nc)),
            ('input_label1', util.tensor2label(self.input_label1, self.opt.label_nc)),
            ('input_label2', util.tensor2label(self.input_label2, self.opt.label_nc)),
            ('predict_label', util.tensor2label(self.predict_label, self.opt.label_nc)),
            ('ori_predict_label', util.tensor2label(self.ori_predict_label, self.opt.label_nc)),
            ('target_label', util.tensor2label(self.target_label, self.opt.label_nc)),
            ])

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def delete_model(self, which_epoch):
        self.delete_network('G', which_epoch, self.gpu_ids)
        self.delete_network('D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
