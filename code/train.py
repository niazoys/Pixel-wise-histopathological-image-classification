import os
import torch
import utils
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch import optim
import torch.nn as nn
from model import uNet19
import distutils.dir_util
from dataset import hist_dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp


def train(net,output,device,epochs,batch,lr,input_shape):
 
    train_dataset = hist_dataset("C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/new_data/trainingdata/",training=True,shape=input_shape)

    val_dataset = hist_dataset("C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/new_data/validationdata/",training=False,shape=input_shape) 

    n_train, n_val = len(train_dataset), len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True,num_workers=1)
  
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True,num_workers=1)
   
    optimizer = optim.Adam(net.parameters(), lr=1e-5*lr)
    
    criterion = nn.CrossEntropyLoss().to(device)

    criterion_dice = utils.DiceLoss()
    
    logging.info("Training Set lenghth: "+str(n_train)+" Validation Set length: "+str(n_val))
    logging.info("Learning Rate: "+str(1e-5*lr))
    logging.info("Batch Size: "+str(batch))
    logging.info("Epochs: "+str(epochs))
    n_batch_train=n_train/batch
    n_batch_val=n_val/batch
    train_acc, val_acc,train_loss,val_loss,train_dice,val_dice= [], [],[], [],[],[]
    best_accuracy = -1
    best_dice_score=-1
    for epoch in range(epochs):
        logging.info(f'epoch {epoch + 1}/{epochs}')
        net.train()
        epoch_loss = 0
        total_train = 0
        correct_train = 0
        dice_score=0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks = batch[0], batch[1]
                imgs = (imgs.to(device=device, dtype=torch.float32))
                masks = (masks.to(device=device, dtype=torch.long))
                preds = net(imgs) 
                loss = criterion(preds, masks)
                dice_loss=criterion_dice(preds, masks)
                loss=loss + dice_loss
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                dice_score+=utils.dice_score(preds, masks).detach().cpu().numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                
                _, preds = torch.max(preds.data, 1)
                total_train += masks.nelement()
                correct_train += preds.eq(masks.data).sum().item()
                
        #Training matrices
        train_accuracy = 100 * (correct_train / total_train)
        train_acc.append(train_accuracy)
        train_loss.append(epoch_loss/n_batch_train)
        train_dice.append(dice_score/(n_batch_train))
        logging.info(f'Training Accuracy: {train_accuracy} Training dice score: {dice_score/n_batch_train} Avg batch loss: {epoch_loss/n_batch_train}')

        #Training matrices
        val_accuracy,validation_loss,val_dice_score= utils.eval_net(net,val_loader,criterion,criterion_dice,device)
        val_acc.append(val_accuracy)
        val_loss.append(validation_loss/n_batch_val)
        val_dice.append(val_dice_score/n_batch_val)
        logging.info(f'Validation Accuracy: {val_accuracy} Validation dice score: {val_dice_score/n_batch_val} Avg Validation batch loss: {validation_loss/n_batch_val}')

        if epoch>0:
            if val_acc[-1]>best_accuracy and val_dice[-1]>best_dice_score:
                os.remove(output + f'epoch_unet16.pth')
                torch.save(net.state_dict(), output + f'epoch_unet16.pth')
                best_accuracy = val_acc[-1]
                best_dice_score=val_dice[-1]
                logging.info(f'checkpoint {epoch} saved !')
        else:
            torch.save(net.state_dict(), output + f'epoch_unet16.pth')
    
        #Save the model and training,validation Accuracy,dice score
        np.save(output+'traning_acc.npy',train_acc)
        np.save(output+'val_acc.npy',val_acc)
        np.save(output+'training_dice.npy',train_dice)
        np.save(output+'val_dice.npy',val_dice)
        np.save(output+'training_loss.npy',train_loss)
        np.save(output+'val_loss.npy',val_loss)

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-o', '--output', type=str, default='C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/results/unet_resnet34_ce_dice_/', dest='output')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, dest='epochs')
    
    parser.add_argument('-b', '--batch', type=int, default=10, dest='batch')
    
    parser.add_argument('-l', '--learning', type=int, default=10, dest='lr')
    
    parser.add_argument('-n', '--network', type=str, default='resnet34', dest='net')
    
    parser.add_argument('-is', '--in_shape', type=int, default=256, dest='in_shape')
    
    parser.add_argument('-f', '--load', type=str, default=None, dest='load')  
        
    return parser.parse_args()

if __name__ == '__main__':
    
    # Setup the logging level
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # parse arguments
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Using device'+str(device))
   
    if args.net=='resnet34': 
        net = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=3,classes=4)
    elif args.net=='vgg':
        net=uNet19(n_classes=4,pretrained=True)
    else:
        logging.info('Network not supported.')

    # make output directory if not exist
    if not os.path.isdir(args.output):
        distutils.dir_util.mkpath(args.output)
    
    # Load previously saved weight
    if args.load != None:        
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    # Transfer the model to cuda 
    net.to(device=device)
    
    # Call trainer method        
    train(net = net,output = args.output,device = device,epochs = args.epochs,batch = args.batch,lr = args.lr,input_shape=args.in_shape)
        
