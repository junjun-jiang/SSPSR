import argparse
import os
import sys
import random
import time
import torch
import cv2
import math
import numpy as np
import torch.backends.cudnn as cudnn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchnet import meter
import utils
import json

from data import HSTrainingData
from data import HSTestData
from SSPSR import SSPSR
from common import *

# loss
from loss import HybridLoss
# from loss import HyLapLoss
from metrics import quality_assessment

# global settings
resume = False
log_interval = 50
model_name = ''
test_data_dir = ''

def main():
    # parsers
    main_parser = argparse.ArgumentParser(description="parser for SR network")
    subparsers = main_parser.add_subparsers(title="subcommands", dest="subcommand")
    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_parser.add_argument("--cuda", type=int, required=False,default=1,
                              help="set it to 1 for running on GPU, 0 for CPU")
    train_parser.add_argument("--batch_size", type=int, default=32, help="batch size, default set to 64")
    train_parser.add_argument("--epochs", type=int, default=40, help="epochs, default set to 20")
    train_parser.add_argument("--n_feats", type=int, default=256, help="n_feats, default set to 256")
    train_parser.add_argument("--n_blocks", type=int, default=3, help="n_blocks, default set to 6")
    train_parser.add_argument("--n_subs", type=int, default=8, help="n_subs, default set to 8")
    train_parser.add_argument("--n_ovls", type=int, default=2, help="n_ovls, default set to 1")
    train_parser.add_argument("--n_scale", type=int, default=4, help="n_scale, default set to 2")
    train_parser.add_argument("--use_share", type=bool, default=True, help="f_share, default set to 1")
    train_parser.add_argument("--dataset_name", type=str, default="Chikusei", help="dataset_name, default set to dataset_name")
    train_parser.add_argument("--model_title", type=str, default="SSPSR", help="model_title, default set to model_title")
    train_parser.add_argument("--seed", type=int, default=3000, help="start seed for model")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                              help="learning rate, default set to 1e-4")
    train_parser.add_argument("--weight_decay", type=float, default=0, help="weight decay, default set to 0")
    train_parser.add_argument("--save_dir", type=str, default="./trained_model/",
                              help="directory for saving trained models, default is trained_model folder")
    train_parser.add_argument("--gpus", type=str, default="1", help="gpu ids (default: 7)")

    test_parser = subparsers.add_parser("test", help="parser for testing arguments")
    test_parser.add_argument("--cuda", type=int, required=False,default=1,
                             help="set it to 1 for running on GPU, 0 for CPU")
    test_parser.add_argument("--gpus", type=str, default="0,1", help="gpu ids (default: 7)")
    # test_parser.add_argument("--test_dir", type=str, required=True, help="directory of testset")
    # test_parser.add_argument("--model_dir", type=str, required=True, help="directory of trained model")

    args = main_parser.parse_args()
    print(args.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.subcommand is None:
        print("ERROR: specify either train or test")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    if args.subcommand == "train":
        train(args)
    else:
        test(args)
    pass

def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    # args.seed = random.randint(1, 10000)
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    print('===> Loading datasets')
    train_path    = './dataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/trains/'
    eval_path     = './dataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/evals/'
    result_path   = './dataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/tests/'
    test_data_dir = './dataset/'+args.dataset_name+'_x'+str(args.n_scale)+'/'+args.dataset_name+'_test.mat'
    
    train_set = HSTrainingData(image_dir=train_path, augment=True)
    eval_set = HSTrainingData(image_dir=eval_path, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=4, shuffle=False)


    if args.dataset_name=='Cave':
        colors = 31
    elif args.dataset_name=='Pavia':
        colors = 102
    else:
        colors = 128    

    print('===> Building model')
    net = SSPSR(n_subs=args.n_subs, n_ovls=args.n_ovls, n_colors=colors, n_blocks=args.n_blocks, n_feats=args.n_feats, n_scale=args.n_scale, res_scale=0.1, use_share=args.use_share, conv=default_conv)
    # print(net)  
    model_title = args.dataset_name + "_" + args.model_title +'_Blocks='+str(args.n_blocks)+'_Subs'+str(args.n_subs)+'_Ovls'+str(args.n_ovls)+'_Feats='+str(args.n_feats)
    model_name = './checkpoints/' + model_title + "_ckpt_epoch_" + str(40) + ".pth"
    args.model_title = model_title
    
    if torch.cuda.device_count() > 1:
        print("===> Let's use", torch.cuda.device_count(), "GPUs.")
        net = torch.nn.DataParallel(net)
    start_epoch = 0
    if resume:
        if os.path.isfile(model_name):
            print("=> loading checkpoint '{}'".format(model_name))
            checkpoint = torch.load(model_name)
            start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(model_name))
    net.to(device).train()

    # loss functions to choose
    # mse_loss = torch.nn.MSELoss()
    h_loss = HybridLoss(spatial_tv=True, spectral_tv=True)
    # hylap_loss = HyLapLoss(spatial_tv=False, spectral_tv=True)
    L1_loss = torch.nn.L1Loss()

    print("===> Setting optimizer and logger")
    # add L2 regularization
    optimizer = Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    epoch_meter = meter.AverageValueMeter()
    writer = SummaryWriter('runs/'+model_title+'_'+str(time.ctime()))
    
    print('===> Start training')
    for e in range(start_epoch, args.epochs):
        adjust_learning_rate(args.learning_rate, optimizer, e+1)
        epoch_meter.reset()
        print("Start epoch {}, learning rate = {}".format(e + 1, optimizer.param_groups[0]["lr"]))
        for iteration, (x, lms, gt) in enumerate(train_loader):
            x, lms, gt = x.to(device), lms.to(device), gt.to(device)
            optimizer.zero_grad()       
            y = net(x, lms)
            loss = h_loss(y, gt)
            epoch_meter.add(loss.item())
            loss.backward()
            # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
            optimizer.step()
            # tensorboard visualization
            if (iteration + log_interval) % log_interval == 0:
                print("===> {} B{} Sub{} Fea{} GPU{}\tEpoch[{}]({}/{}): Loss: {:.6f}".format(time.ctime(), args.n_blocks, args.n_subs, args.n_feats, args.gpus, e+1, iteration + 1,
                                                                   len(train_loader), loss.item()))
                n_iter = e * len(train_loader) + iteration + 1
                writer.add_scalar('scalar/train_loss', loss, n_iter)

        print("===> {}\tEpoch {} Training Complete: Avg. Loss: {:.6f}".format(time.ctime(), e+1, epoch_meter.value()[0]))
        # run validation set every epoch
        eval_loss = validate(args, eval_loader, net, L1_loss)
        # tensorboard visualization
        writer.add_scalar('scalar/avg_epoch_loss', epoch_meter.value()[0], e + 1)
        writer.add_scalar('scalar/avg_validation_loss', eval_loss, e + 1)
        # save model weights at checkpoints every 10 epochs
        if (e + 1) % 5 == 0:
            save_checkpoint(args, net, e+1)

    # save model after training
    net.eval().cpu()
    save_model_filename = model_title + "_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_') + ".pth"
    save_model_path = os.path.join(args.save_dir, save_model_filename)
    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), save_model_path)
    else:
        torch.save(net.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)


    ## Save the testing results
    print("Running testset")
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    net.eval().cuda()
    with torch.no_grad():
        output = []
        test_number = 0
        for i, (ms, lms, gt) in enumerate(test_loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            y = net(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:] 
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = model_title+'_'+str(time.ctime())+ ".txt"
    json.dump(indices, open(QIstr, 'w'))

def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    # switch to evaluate mode
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
        for i, (ms, lms, gt) in enumerate(loader):
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            # y = model(ms)            
            y = model(ms, lms)
            loss = criterion(y, gt)
            epoch_meter.add(loss.item())
        mesg = "===> {}\tEpoch evaluation Complete: Avg. Loss: {:.6f}".format(time.ctime(), epoch_meter.value()[0])
        print(mesg)
    # back to training mode
    model.train()
    return epoch_meter.value()[0]

def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print('===> Loading testset')
    test_set = HSTestData(test_data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('===> Start testing')
    with torch.no_grad():
        epoch_meter = meter.AverageValueMeter()
        epoch_meter.reset()
        # loading model
        model = SSPSR(n_subs=n_subs, n_ovls=n_ovls, n_colors=colors, n_blocks=n_blocks, n_feats=n_feats, n_scale=n_scale, res_scale=0.1, use_share=True, conv=default_conv)
        state_dict = torch.load(model_name)
        model.load_state_dict(state_dict)
        model.to(device).eval()
        mse_loss = torch.nn.MSELoss()
        output = torch.tensor([])
        for i, (ms, lms, gt) in enumerate(test_loader):
            # compute output
            ms, lms, gt = ms.to(device), lms.to(device), gt.to(device)
            # y = model(ms)
            y = model(ms, lms)
            y, gt = y.squeeze().cpu().numpy().transpose(1, 2, 0), gt.squeeze().cpu().numpy().transpose(1, 2, 0)
            y = y[:gt.shape[0],:gt.shape[1],:] 
            if i==0:
                indices = quality_assessment(gt, y, data_range=1., ratio=4)
            else:
                indices = sum_dict(indices, quality_assessment(gt, y, data_range=1., ratio=4))
            output.append(y)
            test_number += 1
        for index in indices:
            indices[index] = indices[index] / test_number

    # save_dir = "/data/test.npy"
    save_dir = result_path + model_title + '.npy'
    np.save(save_dir, output)
    print("Test finished, test results saved to .npy file at ", save_dir)
    print(indices)

    QIstr = model_title+'_'+str(time.ctime())+ ".txt"
    json.dump(indices, open(QIstr, 'w'))

def save_checkpoint(args, model, epoch):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    checkpoint_model_dir = './checkpoints/'
    if not os.path.exists(checkpoint_model_dir):
        os.makedirs(checkpoint_model_dir)
    ckpt_model_filename = args.dataset_name + "_" + args.model_title + "_ckpt_epoch_" + str(epoch) + ".pth"
    ckpt_model_path = os.path.join(checkpoint_model_dir, ckpt_model_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, ckpt_model_path)
    model.to(device).train()
    print("Checkpoint saved to {}".format(ckpt_model_path))


if __name__ == "__main__":
    main()
