from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import elasticdeform
import numpy as np
import cv2
from torchvision.utils import save_image
import random

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image
    
    
class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):    
        image = (image - self.mean) / self.std
        return image

def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(), ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class BrainMRIDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform
        self.csf_transform = transforms.Compose([transforms.ToTensor()])
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.csf_paths = self.load_csf_dataset()

        self.mu = 181.9980541123061
        self.sigma = 21.06710470152199

    def load_dataset(self):
        img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        return img_paths
    
    def load_csf_dataset(self):
        csf_paths = glob.glob(os.path.join(self.img_path, 'csf') + "/*.png")
        return csf_paths

    def __len__(self):
        return len(self.img_paths)

    def add_ellipse_gaussian_noise(self, image, center_x, center_y, radius_x, radius_y, sigma=0.9, point=10, csf=None):
        height, width, _ = image.shape
    
        x, y = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate distance from the center for each pixel
        distance = np.sqrt((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2)

        # Create an ellipse mask
        ellipse_mask = np.where(distance <= 1, 1, 0)
        
        ellipse_mask = elasticdeform.deform_random_grid(ellipse_mask, sigma=sigma, points=point, order=0)
        ellipse_mask = ellipse_mask * (np.mean(image, axis=2) != 0)

        # Intensity
        noise = np.random.normal(self.mu, self.sigma, size=(height, width))
        noise = np.stack([noise]*3, axis=-1) 
        noise = (noise - self.mu) / (4 * self.sigma) + 0.5

        # Apply the noise only to the region within the ellipse mask
        image_with_noise = np.where(ellipse_mask[..., np.newaxis], noise, image)
        
        # Set csf's ellipse_mask region to 0
        csf[0, ellipse_mask == 1] = 0
        
        return image_with_noise, csf, ellipse_mask
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        csf_path = self.csf_paths[idx]
        csfs = cv2.imread(csf_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img/255., (256,256))
        img_normal = self.transform(img)
        csf = self.csf_transform(csfs)
        
        roi_path = os.path.join(self.img_path + "/roi/ory.nii._{}.png".format(img_path.split(".png")[0].split("_")[1]))
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        # select lobe
        if 0 <= int(img_path.split(".png")[0].split("_")[1]) <= 15:
            value = random.choice([102, 148, 194, 235])
        elif 16 <= int(img_path.split(".png")[0].split("_")[1]) <= 40:
            value = random.choice([102, 148, 194, 235, 255])
        elif 41 <= int(img_path.split(".png")[0].split("_")[1]) <= 42:
            value = random.choice([102, 148, 235, 255])
        elif 43 <= int(img_path.split(".png")[0].split("_")[1]) <= 49:
            value = random.choice([102, 235, 255])

        # Generate noise parameters
        indices = np.argwhere(roi == value)

        random_index = random.choice(indices)
        center_x, center_y = random_index

        radius_x, radius_y = np.random.randint(20, 35), np.random.randint(20, 35)
        sigma, point = np.random.randint(1, 8), np.random.randint(20,30)

        img_noise, csf_image, tumor_mask = self.add_ellipse_gaussian_noise(img, center_x, center_y, radius_x, radius_y, sigma=sigma, point=point, csf=csf)

        img_noise = self.transform(img_noise)

        return img_normal, img_noise, csf_image, tumor_mask, img_path.split('/')[-1]


class BrainMRIDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.csf_transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
       
        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img = self.transform(img)
        
        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], "image.size != gt.size !!!"

        return (img, gt, label, img_type, img_path.split('/')[-1])

