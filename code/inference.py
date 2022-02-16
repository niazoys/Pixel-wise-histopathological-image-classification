import os
import torch
import utils
import logging
import argparse
import numpy as np
from model import uNet19
import distutils.dir_util
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import hist_dataset_test,hist_dataset

def train(net,output,device,batch):
 
    test_dataset = hist_dataset_test("C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/new_data/testdata/")
    
    val_dataset = hist_dataset("C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/new_data/validationdata/",training=False,shape=256) 

    ex_test_dataset = hist_dataset("C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/new_data/external_test/",training=False,shape=256) 

    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    ex_test_loader = DataLoader(ex_test_dataset, batch_size=batch, shuffle=False)

    
    l,n=[],0
    net.eval()
    for batch in val_loader:
        imgs,gt,original_image=batch[0],batch[1],batch[2]
        imgs =(imgs.to(device=device, dtype=torch.float32))
        with torch.no_grad():
            preds = net(imgs)
        preds=preds.cpu()
        # accumulate the stats for each batch
        l.append(utils.stats(preds,gt))
        _, preds = torch.max(preds.data, 1)

        #Generate the write_overlayed_image
        utils.write_overlayed_image(original_image=original_image.cpu().numpy(),preds=preds.cpu().numpy(),out=output,batch=n,groundtruth=gt.cpu().numpy())   
        n+=1
    
    # Average all batch stats
    l=np.mean(np.array(l),axis=0)

    
    #Save the stats
    file = open(output+'stat.txt','w')
    file.write('Precision = %f\n'%(l[0]))
    file.write('Recall = %f\n'%(l[1]))
    file.write('Dice = %f\n'%(l[2]))
    file.close()

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--batch', type=int, default=5, dest='batch')
    
    parser.add_argument('-n', '--network', type=str, default='resnet34', dest='net')
    
    parser.add_argument('-f', '--load', type=str, default='C:/Users/Niazo/OneDrive/Desktop/PhDTask.v1/PhDTask/results/unet_resnet34_ce_dice/', dest='load')  
        
    return parser.parse_args()

if __name__ == '__main__':
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info('Using device'+str(device))
   
    if args.net=='resnet34': 
        net = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=3,classes=4)
    elif args.net=='vgg':
        net=uNet19(n_classes=4,pretrained=True)
    else:
        logging.info('Network not supported.')

    #laod weights
    if args.load != None:        
        net.load_state_dict(torch.load(args.load+'epoch_unet16.pth', map_location=device))
        logging.info(f'Model loaded from {args.load}')
    
    # create directory if not already exists
    output =args.load+'val_mask/'
    if not os.path.exists(output):
        distutils.dir_util.mkpath(output)

    net.to(device=device)
            
    train(net = net,output = output,device = device,batch = args.batch)
        
