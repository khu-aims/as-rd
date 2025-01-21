import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj, show_cam_on_image
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import BrainMRIDataset_test, get_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default = './your_checkpoint_folder', type=str)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--classes', nargs="+", default=["upenn"])
    pars = parser.parse_args()
    return pars

def inference(_class_, pars):
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    test_path = './CamCAN/' + _class_

    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    test_data = BrainMRIDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
  
    auroc_px, auroc_sp, aupro_px, dice, sample_accuracy, sample_f1, sample_precision, sample_recall, pixel_precision, pixel_recall = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)
    print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}, Dice: {:.4f}'.format(_class_, auroc_sp, auroc_px, aupro_px, dice))
    print('{}: Sample Accuracy: {:.4f}, Sample Precision:{:.4f}, Sample Recall: {:.4f}, Sample F1score: {:.4f}'.format(_class_, sample_accuracy, sample_precision, sample_recall, sample_f1))
    print('{}: Sample Precision:{:.4f}, Sample Recall: {:.4f}'.format(_class_, pixel_precision, pixel_recall))
    return auroc_sp, auroc_px, aupro_px, dice, sample_accuracy, sample_f1, sample_precision, sample_recall, pixel_precision, pixel_recall


if __name__ == '__main__':
    pars = get_args()

    item_list = [ 'upenn', 'brats', 'lumiere' ]
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': [], 'Dice': []}
    
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px, dice, sample_accuracy, sample_f1, sample_precision, sample_recall, pixel_precision, pixel_recall = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        metrics['Dice'].append(dice)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'{pars.checkpoint_folder}/metrics_checkpoints.csv', index=False)