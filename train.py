import argparse
import pickle
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import get_dataset, get_model, get_transforms, set_seed,\
    get_optimizer_and_scheduler, get_coreset, eval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--data", type=str, default="data", help="path to save/load data")
    parser.add_argument("--dataset", type=str, default="CIFAR100", help="dataset to train on")
    parser.add_argument("--epoch", type=int, default=200, help="train epochs")
    parser.add_argument("--eval_freq", default=2000, help="evaluate frequency during training")
    parser.add_argument("--model_dir", default="./ckpt", help="dir to save model")
    parser.add_argument("--lr", default=0.1, help="learning rate", type=float)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--prune", type=str, default=None, help="path to pruned image ids")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default=True, type=bool, help="whether to save model checkpoint or not")
    parser.add_argument("--arch", default="resnet50", help="backbone architecture")
    args = parser.parse_args()
    return args


def main(args):
    
    device = args.device
    save_dir = os.path.join(args.model_dir, args.dataset, args.arch, "")
    os.makedirs(save_dir, exist_ok=True)
    
    train_transform = get_transforms(args, train=True)
    test_transform = get_transforms(args, train=False)
    
    if args.prune is not None:
        with open(args.prune, "rb") as f:
            drop_id = pickle.load(f)
        
        data_train = get_coreset(args, transform=train_transform, drop_id=drop_id)
        # Don't save model checkpoint when it is trained on coreset
        args.save = False
    else:
        # train on normal dataset
        data_train = get_dataset(args, transform=train_transform, train=True)
        
    data_test = get_dataset(args, transform=test_transform, train=False)        
    trainloader = DataLoader(data_train, batch_size=args.batch_size, pin_memory=True, num_workers=5, shuffle=True)
    testloader = DataLoader(data_test, batch_size=32, pin_memory=True, num_workers=4)
    
    print("trainset size:", len(trainloader.dataset))
    print("Starting at ",time.asctime( time.localtime(time.time()) ))
    
    model = get_model(args)
    model = model.to(args.device)
    optimizer, scheduler = get_optimizer_and_scheduler(args, model)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(args.epoch):
        model.train()
        losses, accs = [], []
        for i, (_, img, target) in enumerate(trainloader):
            img, target = img.to(device), target.to(device)
            output = model(img)
            loss = criterion(output, target)
            pred = output.argmax(1)
            acc = torch.eq(pred, target).float().mean().cpu().numpy()
                        
            optimizer.zero_grad()
            loss.backward()
            
            if "tiny" in args.data:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            losses.append(loss.item())
            accs.append(acc.item())
            
        
        train_loss, train_acc = np.mean(losses), np.mean(accs)
        eval_loss, eval_acc = eval(model, testloader, criterion, device)
        print("Epoch: {}, train loss: {:.4f}, train accs: {:.4f}, "
            "val loss: {:.4f}, val acc: {:.4f}".format(epoch, train_loss, train_acc*100, eval_loss, eval_acc*100))
        
        if best_acc < eval_acc:
            best_acc = eval_acc
            if args.save:
                save_path = os.path.join(save_dir, "best.pth")
                torch.save(model.state_dict(), save_path)
        scheduler.step()
        
    if args.save:
        save_path = os.path.join(args.save_dir, "last.pth")
        torch.save(model.state_dict(), save_path)
        
    print("Ended at ", time.asctime( time.localtime(time.time()) ))
    print("Best accuracy: ", best_acc)
    
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)