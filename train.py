import os
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
import dataset.dataset as dataset

from models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=777, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

parser.add_argument('--model-architecture', default='whitenet', type=str)
                    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item() # [0, batch_size]
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy
        

def train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration):
    global device

    model.train()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    t1 = time.time()
    for i, (input, target) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)

            pred = model(input)

            loss = criterion(pred, target)

        loss_batch.append(loss.item())
        
        top1_accuracy, top5_accuracy = accuracy(pred, target)
        
        top1_accuracy_batch.append(top1_accuracy)
        top5_accuracy_batch.append(top5_accuracy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 50 == 0:

            torch.cuda.synchronize()
            t2 = time.time()
            
            print("epoch: ", epoch,
             "iteration: ", i, "/", total_iteration,
             " loss: ", np.mean(loss_batch),
             " top1 acc: ", np.mean(top1_accuracy_batch),
             " top5 acc: ", np.mean(top5_accuracy_batch),
             " time per 50 iter(sec): ", t2 - t1)

            t1 = time.time()
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    with torch.no_grad():
        for i, (input, target) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            
            pred = model(input)
            loss = criterion(pred, target)
            loss_batch.append(loss.item())
            
            top1_accuracy, top5_accuracy = accuracy(pred, target) 
        
            top1_accuracy_batch.append(top1_accuracy)
            top5_accuracy_batch.append(top5_accuracy)
            
        
    loss = np.mean(loss_batch)
    top1_accuracy = np.mean(top1_accuracy_batch)
    top5_accuracy = np.mean(top5_accuracy_batch)
    
    print("val-", "epoch: ", epoch, " loss: ", loss, " top1 acc: ", top1_accuracy, " top5 acc: ", top5_accuracy)
    
    return top1_accuracy

if __name__ == '__main__':
    args = parser.parse_args()

    set_random_seed(args.seed)
    
    train_dataset = dataset.ImageClassificationDataset(dataset_path='D:/datasets/ILSVRC2012_ImageNet/ILSVRC2012_img_train', phase="train")
    validation_dataset = dataset.ImageClassificationDataset(dataset_path='D:/datasets/ILSVRC2012_ImageNet/ILSVRC2012_img_val', phase="validation")
    
    train_dataset_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    model_dict = {'whitenet': whitenet.WhiteNet(),
                  'tiny': tiny.YOLOv3TinyBackbone()}

    model = model_dict[args.model_architecture]
    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    print("len of train_dataset: ", len(train_dataset))
    print("len of validation_dataset: ", len(validation_dataset))
    
    start_epoch = 0
    best_top1_accuracy = 0.

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weight)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        top1_accuracy = checkpoint['top1_accuracy']
        best_top1_accuracy = checkpoint['best_top1_accuracy']
    
    print("#parameters of model: ", utils.count_total_prameters(model))
    
    total_iteration = len(train_dataset)//args.batch_size

    for epoch in range(start_epoch, args.epochs):
        train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration)
        top1_accuracy = validation(model, criterion, optimizer, validation_dataset_loader, epoch)
        scheduler.step()
            
        state = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scaler_state_dict' : scaler.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'top1_accuracy': top1_accuracy,
        'best_top1_accuracy': max(best_top1_accuracy, top1_accuracy)
        }
    
        torch.save(state, os.path.join("./", args.model_architecture+"_latest.pth"))

        if best_top1_accuracy <= top1_accuracy:
            best_top1_accuracy = top1_accuracy
            torch.save(state, os.path.join("./", args.model_architecture+"_best.pth"))     