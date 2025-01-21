import torch
import torch.nn as nn
import numpy as np
import random
import cv2
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
from PIL import Image
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from utils.utils_test import evaluation_multi_proj, train_cal_anomaly_map
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion, CsfLoss, NoiseDistillationLoss
from dataset.dataset import BrainMRIDataset_test, BrainMRIDataset_train, get_data_transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default = './checkpoint_result', type=str)
    parser.add_argument('--batch_size', default = 16, type=int)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--detail_training', default='note', type = str)
    parser.add_argument('--proj_lr', default = 0.001, type=float)
    parser.add_argument('--distill_lr', default = 0.005, type=float)
    parser.add_argument('--weight_proj', default = 0.2, type=float) 
    parser.add_argument('--classes', nargs="+", default=["upenn"])
    pars = parser.parse_args()
    return pars

def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    train_path = './CamCAN/' + _class_ + '/train'
    test_path = './CamCAN/' + _class_
    
    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path  = pars.save_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    train_data = BrainMRIDataset_train(root=train_path, transform=data_transform)
    test_data = BrainMRIDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained ImageNet for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss()
    csf_loss = CsfLoss()
    noise_loss = NoiseDistillationLoss()

    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=pars.proj_lr, betas=(0.5,0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=pars.distill_lr, betas=(0.5,0.999))


    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0
    
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    
    loss_proj = []
    loss_distill = []
    total_loss = []
    
    history_infor = {}

    # set appropriate epochs for specific classes
    if _class_ in ['tumor']:
        num_epoch = 6

    print(f'with class {_class_}, Training with {num_epoch} Epoch')
    
    for epoch in tqdm(range(1,num_epoch+1)):
        bn.train()
        proj_layer.train()
        decoder.train()
        loss_proj_running = 0
        loss_distill_running = 0
        total_loss_running = 0
        
        ## gradient acc
        accumulation_steps = 2
        
        for i, (img, img_noise, csf, tumor_mask, _) in enumerate(train_dataloader):
            img = img.to(device).float()
            img_noise = img_noise.to(device).float()

            inputs = encoder(img)
            inputs_noise = encoder(img_noise)
            csf = csf.to(device)
            (feature_space_noise, feature_space) = proj_layer(inputs, features_noise = inputs_noise)

            outputs = decoder(bn(feature_space)) #bn(inputs))
            noise_outputs = decoder(bn(feature_space_noise))


            anomaly_map, _ = train_cal_anomaly_map(inputs_noise, noise_outputs, device, img.shape[-1], amap_mode='gh')

            L_csf = csf_loss(anomaly_map.to(torch.float32), csf.to(torch.float32))
            L_noise_distill = noise_loss(anomaly_map.to(torch.float32), tumor_mask.to(torch.float32))
            L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)
            L_distill = loss_fucntion(inputs,outputs)
            loss = L_distill + 0.01 * (L_csf + L_noise_distill) + pars.weight_proj * L_proj
            optimizer_proj.zero_grad()
            optimizer_distill.zero_grad()

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_proj.step()
                optimizer_distill.step()

            total_loss_running += loss.detach().cpu().item()
            loss_proj_running += L_proj.detach().cpu().item()
            loss_distill_running += L_distill.detach().cpu().item()
            
        auroc_px, auroc_sp, aupro_px, sample_accuracy, sample_f1, sample_precision, sample_recall, pixel_precision, pixel_recall = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)        
        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)
        loss_proj.append(loss_proj_running)
        loss_distill.append(loss_distill_running)
        total_loss.append(total_loss_running)
        
        figure = plt.gcf() # get current figure
        figure.set_size_inches(8, 12)
        fig, ax = plt.subplots(3,2, figsize = (8, 12))
        ax[0][0].plot(auroc_px_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list)
        ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list)
        ax[1][0].set_title('aupro_px')
        ax[1][1].plot(loss_proj)
        ax[1][1].set_title('loss_proj')
        ax[2][0].plot(loss_distill)
        ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss)
        ax[2][1].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_traning.png', dpi = 100)
        
        print('Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(epoch, auroc_sp, auroc_px, aupro_px))

        if (auroc_px + auroc_sp) / 2 > best_score:
            best_score = (auroc_px + auroc_sp ) / 2
            
            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({'proj': proj_layer.state_dict(),
                       'decoder': decoder.state_dict(),
                        'bn':bn.state_dict()}, save_model_path)

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px

if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)
    all_classes = ['upenn']
    setup_seed(120)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f} '.format(c, auroc_sp, auroc_px, aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)

        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)