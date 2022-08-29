import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import models
import os

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image



def saliency_map(image):
    # args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    # model = models.resnet18(pretrained=True)
    model = models.vgg16(pretrained=True)
    # model = models.inception_v3(pretrained=True)
    # print(model)
    # print(model.Mixed_7c.branch_pool)
    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    model.cuda()
    # target_layer = model.layer4[-1]
    # target_layer = model.Mixed_7c.branch_pool
    target_layer = model.features[-1]
    cam = methods['gradcam++'](model=model,
                               target_layer=target_layer,
                               use_cuda=False)

    input_tensor = image
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 1

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        aug_smooth=False,
                        eigen_smooth=False)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam