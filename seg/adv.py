import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Function
import math
from torchvision import transforms
import logging
from .segment import parse_args, DRNSeg, SegList, SegListSeg, SegListSegAdv
import os.path as osp
import json
import numpy as np
import drn
from PIL import Image
from PIL import ImageFilter
from random import randint
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from pdb import set_trace as st
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)
args = parse_args()

writer = SummaryWriter(log_dir= "runs/blur_{}".format(args.K))

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

#test

def transform(image):
    mean = [0.29010095242892997, 0.32808144844279574, 0.28696394422942517]
    std = [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]
    image[:, 0, :, :] = ( image[:, 0, :, :] - mean[0] ) / std[0]
    image[:, 1, :, :] = ( image[:, 1, :, :] - mean[1]) / std[1]
    image[:, 2, :, :] = (image[:, 2, :, :] - mean[2]) / std[2]
    return image

def attack(args, net, loss, data_x, target, points = [], target_points = []):
    x = data_x.clone()
    noise = torch.zeros(x.size())
    mask = torch.zeros_like(noise)
    mask_target = torch.zeros_like(target)
    point1 = int(points[0])
    point2 = int(points[1])
    point3 = int(points[2])
    point4 = int(points[3])
    target_point1 = int(target_points[0])
    target_point2 = int(target_points[1])
    target_point3 = int(target_points[2])
    target_point4 = int(target_points[3])
    if args.use_cuda:
        x = x.cuda()
        target = target.cuda()
        noise = noise.cuda()
        mask = mask.cuda()
        mask_target = mask_target.cuda()
    mask[:,:,point2:point4,point1:point3] = 1.0
    mask_target[:,target_point2:target_point4,target_point1:target_point3] = 1.0
    mask_logits = torch.unsqueeze(mask_target,1).repeat(1,19,1,1)
    # init_noise = torch.zeros_like(mask).uniform_(0.0, 1.0)
    # noise[:,:,point2:point4,point1:point3] = init_noise[:,:,point2:point4,point1:point3]
    x = Variable(x, requires_grad=False)
    noise = Variable(noise, requires_grad=True)
    mean_var = Variable(mean, requires_grad=True)
    std_dv_var = Variable(std_dv, requires_grad=True)
    optimizer = optim.Adam([noise], lr=1e-2)
    # Compute the range of noise
    entropy_loss = torch.nn.NLLLoss2d(ignore_index=255)
    for i in range(50):
        net.zero_grad()
        optimizer.zero_grad()
        loss.zero_grad()
        # noise.zero_grad()
        # mean_var.zero_grad()
        # std_dv_var.zero_grad()
        x_hat = torch.clamp(x + noise*mask, 0.0, 1.0)
        x_normal = (x_hat-mean_var)/std_dv_var
        logits = net(x_normal)[0]
        # hou_loss = -entropy_loss(logits*mask_logits.float(), target*mask_target)
        # hou_loss = -loss(logits*mask_logits.float(), target*mask_target) * 10 + loss(logits*(1-mask_logits.float()), target*(1-mask_target)) / 10
        hou_loss = loss(logits*mask_logits.float(), target*mask_target) * 10
        pred = torch.max(logits, 1)[1]
        print('acc: %.3f' %( (pred == target).cpu().data.numpy().sum() / (1024 * 2048) ))
        print('iteration %d loss %.3f'%(int(i), hou_loss.cpu().data.numpy()))
        hou_loss.backward()
        optimizer.step()
    # st()
    # np.save('tmp_adv.npy', x_hat.cpu().data.numpy())
    # tmp_data = np.load('tmp_adv.npy')
    # tmp_data = transform(tmp_data)
    # tmp_data_var = Variable( torch.from_numpy(tmp_data).cuda(), requires_grad=False)
    # pred = torch.max( net(tmp_data_var), 1)[1]
    # acc = (pred == target).cpu().data.numpy().sum()
    #
    return x_hat.cpu().data
    # return torch.clamp(x+noise, 0.0, 1.0).data.cpu()




def vis_seg(seg_result):
    seg_result = seg_result.cpu().data.numpy()
    if seg_result.ndim == 3:
        seg_result = seg_result[0]
    img_out = np.zeros([seg_result.shape[0], seg_result.shape[1], 3])
    for i in range(int(seg_result.shape[0])):
        for j in range(int(seg_result.shape[1])):
            pred = seg_result[i, j]
            if pred > 18:
                pred = 19
            img_out[i, j] = CITYSCAPE_PALETTE[pred]
    return img_out.astype(np.uint8)

def convert_img(img, noise=False):
    img2 = img.data.cpu().numpy()
    if img2.ndim == 4:
        img2 = img2[0]
    if noise:
        img2 = (img2 + 1 ) /2.0
    img2 = np.transpose(img2, [1, 2, 0]) * 255.0
    return img2

def vis_result(save_path, iteration, original_img, perturbed_img, pred, pred_attack, gt, target, perceptibility):
    save_path = osp.join(save_path, '%d-%.3f.png'%(int(iteration), perceptibility))
    image = Image.new('RGB', (original_img.shape[1]*3, original_img.shape[0]*2))
    image.paste(Image.fromarray(original_img), [0, 0])
    image.paste(Image.fromarray(perturbed_img), [0, original_img.shape[0]])
    image.paste(Image.fromarray(pred), [original_img.shape[1], 0])
    image.paste(Image.fromarray(pred_attack), [original_img.shape[1], original_img.shape[0]])
    image.paste(Image.fromarray(gt), [original_img.shape[1]*2, 0])
    image.paste(Image.fromarray(target), [original_img.shape[1]*2, original_img.shape[0]])
    image.save(save_path)


def vis_results(save_path, iteration, attack_imgs, segmentations):
    save_path = osp.join(save_path, str(iteration)+'.png')
    num_im = len(attack_imgs)
    column_num = int((num_im+1)//2)
    width = attack_imgs[0].shape[1]
    height = attack_imgs[0].shape[0]

    image = Image.new('RGB', (attack_imgs[0].shape[1] * column_num, attack_imgs[0].shape[0] * 4))
    for i in range(column_num):
        image.paste(Image.fromarray(attack_imgs[i]), [width*i, 0])
        image.paste(Image.fromarray(segmentations[i]), [width * i, height])
    for i in range(column_num, num_im):
        image.paste(Image.fromarray(attack_imgs[i]), [width * (i-column_num), height*2])
        image.paste(Image.fromarray(segmentations[i]), [width * (i-column_num), height*3])
    image.save(save_path)

def regular_attack():
    args = parse_args()
    args.use_cuda = torch.cuda.is_available()
    # args.pretrained = './scale2_512/model_best.pth.tar'
    
    # args.pretrained = './scale2_512/model_best.pth.tar'
    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    if args.pretrained:
        single_model.load_state_dict(torch.load(args.pretrained))
    model = torch.nn.DataParallel(single_model)
    if args.use_cuda:
        model = model.cuda()
    phase = args.phase
    data_dir = args.data_dir
    save_dir = args.save_dir + '_regular_100_' + args.phase
    original_dir = osp.join(save_dir, 'original')
    adversarial_dir = osp.join(save_dir, 'adversarial')
    label_dir = osp.join(save_dir, 'label')
    if not osp.exists(save_dir):
        import os
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(osp.join(save_dir, 'original'))
        os.makedirs(osp.join(save_dir, 'adversarial'))
        os.makedirs(osp.join(save_dir, 'label'))
    info = json.load(open(osp.join(data_dir, 'info.json'), 'r'))
    global mean
    mean = torch.FloatTensor(info['mean'])
    mean = mean.view(1, 3, 1, 1)
    global std_dv
    std_dv = torch.FloatTensor(info['std'])
    std_dv = std_dv.view(1, 3, 1, 1)
    if args.use_cuda:
        mean = mean.cuda()
        std_dv = std_dv.cuda()
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    starting = 0
    ending = None
    if args.starting is not None:
        starting = args.starting
    if args.ending is not None:
        ending = args.ending


    # edit start
    dataset = SegListSegAdv(data_dir, phase, transforms.Compose([
        transforms.ToTensor(),
    ]), out_name=True, starting=starting, ending=ending)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False, num_workers=1,
        pin_memory=False
    )



    # edit end

    loss = houdini_loss(args.use_cuda, ignore_index=255)
    if args.use_cuda:
        loss = loss.cuda()
    ## Untargeted Attack


    # edit start
    for iter, (data, label, label1, name, box, target_box) in enumerate(test_loader):
        # pass
        print(int(iter))
        # print(label)
        data_untar = Variable(data)
        if args.use_cuda:
            data_untar = data_untar.cuda()
        predict = model(data_untar)[0]
        # predict_2 = torch.topk(predict, 2, dim=1)
        predict = predict.max(1)[1]
        seg_vis = vis_seg(predict)
        # target = torch.LongTensor(label.size()).zero_()
        # target[label == 255] = 255
        target = label1
        # target = label1
        if args.use_cuda:
            target = target.cuda()
        target = Variable(target)
        target_vis = vis_seg(target)
        points = box[0].split(',')
        target_points = target_box[0].split(',')

        attacker = attack(args, model, loss, data, target, points, target_points)

        attacker_const = attacker.clone()
        attacker = attacker.numpy()
        if attacker.ndim == 4:
            attacker = attacker[0]
        attacker = np.transpose(attacker, [1, 2, 0])
        attacker_var = Variable(torch.from_numpy(transform(attacker[np.newaxis].transpose(0,3,1,2))), requires_grad=False)
        pred = model(attacker_var)[0].max(1)[1]
        acc = (target == pred).cpu().data.numpy().sum() / (1024 * 2048)
        np.save('tmp_adv.npy', attacker)
        # st()
        np.save(osp.join(adversarial_dir, str(int(iter)+starting) + '.npy'), attacker)

        data = data.numpy()
        if data.ndim == 4:
            data = data[0]
        data = np.transpose(data, [1, 2, 0])
        np.save(osp.join(original_dir, str(int(iter)+starting) + '.npy'), data)

        label_const = label.clone()
        label = label.numpy()
        if label.ndim == 3:
            label = label[0]
        np.save(osp.join(label_dir, str(int(iter)+starting) + '.npy'), label)
        attacker_img = convert_img(Variable(attacker_const))
        original_img = convert_img(data_untar)
        ## Calculate the perceptibility
        percetibility = np.sum((attacker_img - original_img)**2)/(attacker_img.shape[0]*attacker_img.shape[1])
        ## Visualize the attacked results
        attacker = Variable(attacker_const)
        if args.use_cuda:
            attacker = attacker.cuda()
        attack_seg = model(attacker)[0]
        attack_seg = attack_seg.max(1)[1]
        attack_seg_vis = vis_seg(attack_seg)
        gt_vis = vis_seg(Variable(label_const))
        original_img = np.uint8(original_img)
        attacker_img = np.uint8(attacker_img)
        vis_result(save_dir, iter, original_img, attacker_img, seg_vis, attack_seg_vis, gt_vis, target_vis, percetibility)
        print(percetibility)


if __name__ == '__main__':
 
    regular_attack()
    # blur_attack()
    # random_attack()