import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import sys
import os

import numpy as np

import argparse
from util.arguments import get_arguments
from util.misc import AverageMeter
from util.loss import CrossEntropyLoss2d
from dataset import get_dataloader

from model.PSPNet import PSPNet

import pdb

writer = SummaryWriter()

def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def train(args,net,train_loader, optimizer, scheduler, criterion, epoch):
    losses = AverageMeter()
    Avgacc = AverageMeter()
    meanIoU = AverageMeter()

    net.train()

    for idx,(images, labels) in enumerate(train_loader):
        images, labels = images.to(args.device), labels.to(args.device)
        labels = labels.long()
        output = net(images)
        
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            label_temp = torch.where(labels == -1, pred, labels)
        
        loss = criterion(output,label_temp)
    

        losses.update(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)

        prediction = F.softmax(output,dim=1)
        
        acc = pixel_acc(prediction,labels)
        Avgacc.update(acc.item())

        # print info
        if idx %1 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
                  'IoU {iou.val:.3f} ({iou.avg:.3f})\t'.format(
                   epoch, idx + 1, len(train_loader),loss=losses, acc=Avgacc, iou=meanIoU))
    sys.stdout.flush()

    writer.add_scalar("Train Loss",losses.avg, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]['lr'])

    return losses.avg, Avgacc.avg, meanIoU.avg

def validate(args,net,val_loader,num_classes,):
    net.eval()
    pred_all = []
    gts_all = []

    for img,gts in val_loader:
        img, gts = img.to(args.device), gts.to(args.device)
        with torch.no_grad():
            output =F.softmax(net(img),1)

        pred = output.data.max(1)[1].cpu().numpy()

        pred_all.append(pred)
        gts_all.append(gts.data.cpu().numpy())
 
    gts_all = np.concatenate(gts_all)
    pred_all = np.concatenate(pred_all)

    acc, acc_cls, mean_iou, fwavacc = evaluate(pred_all, gts_all,num_classes)

    print ('--------------------------------------------------------------------')
    print ('[acc %.5f], [acc_cls %.5f], [mean_iou %.5f], [fwavacc %.5f]' % (
        acc, acc_cls, mean_iou, fwavacc))
    
    return acc, mean_iou

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'args': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def main():
    args = argparse.ArgumentParser()
    args = get_arguments()

    args.num_classes=150

    args.save_folder = './save/'+args.arch

    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda',args.gpu_id)
    
    train_loader, val_loader = get_dataloader(args)
    
    net = PSPNet(args, num_classes = args.num_classes)
    optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9, weight_decay=0.0001)
    lr_lambda = lambda epoch: pow((1-epoch/args.epochs),0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch=0

    if args.resume :
        pdb.set_trace()
        load_path= os.path.join(
                        args.save_folder+'/best'+args.name_opt+'.pth')
        check_point = torch.load(load_path, map_location='cpu')
        start_epoch = check_point['epoch']
        net.load_state_dict(check_point['model'])
        optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.wd)
        optimizer.load_state_dict(check_point['optimizer'])
    
    criterion = CrossEntropyLoss2d()
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    best_acc = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epochs):
        _,_,_ = train(args,net,train_loader,optimizer,scheduler,criterion,epoch)
        acc, mean_iou = validate(args,net,val_loader,num_classes=args.num_classes)

        if acc > best_acc:
            best_acc = acc
            best_iou=mean_iou
            save_file = os.path.join(
                        args.save_folder+'/best'+args.name_opt+'.pth')
            print(save_file)
            save_model(net, optimizer, args, epoch, save_file)
        print("best acc:",best_acc)
        print("best iou:",best_iou)
    writer.close()

if __name__== '__main__' :
    main()