from metrics.FRNet import VGGFace2Loss
from skimage.io import imread
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

def main():
    img_path = 'data/celeba_hq_deca_v2/Original_DECA/val/input/male/000080.jpg'
    img = imread(img_path)

    img_torch = torchvision.transforms.ToTensor()(img.astype(np.float32))
    img_torch = img_torch.view(-1, *img_torch.shape)

    facerec_loss = VGGFace2Loss(unnormalize=True)


    facerec_loss(img_torch, img_torch)




if __name__ == "__main__":
    main()
