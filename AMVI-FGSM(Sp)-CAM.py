#    Authors:    Chao Li, Wen Yao, Handing Wang, Tingsong Jiang
#    Xidian University, China
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    EMAIL:      lichaoedu@126.com
#    DATE:       August 2022
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# Chao Li, Wen Yao, Handing Wang, Tingsong Jiang, Adaptive momentum variance for attention-guided sparse adversarial attacks, Pattern Recognition, 2022.
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------


import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torch.nn as nn
import copy
from cam import saliency_map



root = "/home/amax/"


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        data_root = r'/home/amax/ILSVRC2012_img_val'
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            words[0] = os.path.join(data_root, words[0])
            assert os.path.exists(words[0])
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def mbvmifgsm_attack(model, loss, images, labels, eps, set):
    dim1_min = (0 - 0.485) / 0.229
    dim1_max = (1 - 0.485) / 0.229
    dim2_min = (0 - 0.456) / 0.224
    dim2_max = (1 - 0.456) / 0.224
    dim3_min = (0 - 0.406) / 0.225
    dim3_max = (1 - 0.406) / 0.225
    img = copy.deepcopy(images)
    attack_steps = 5
    alpha = eps / attack_steps
    decay = 1.0
    momentum = torch.zeros_like(images, device=images.device)
    vel2 = torch.zeros_like(images, device=images.device)
    for step in range(attack_steps):
        img = img.cpu().detach().numpy()
        images.requires_grad = True
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        cost = loss(outputs, labels).to(device)
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        grad_c = copy.deepcopy(grad)
        grad_n = torch.zeros_like(images, device=images.device)
        for i in range(20):
            img_neighbor = img + np.random.uniform(-0.15, 0.15, img.shape)
            img_neighbor = torch.from_numpy(img_neighbor)
            img_neighbor.requires_grad = True
            img_neighbor = img_neighbor.to(torch.float32)
            img_neighbor = img_neighbor.to(device)
            output_n = model(img_neighbor)
            cost_n = loss(output_n, labels).to(device)
            grad_nn = torch.autograd.grad(cost_n, img_neighbor, retain_graph=False, create_graph=False)[0]
            grad_n += grad_nn
        grad_n /= 20
        vel = grad_n - grad_c
        p1 = (step + 1) / attack_steps
        grad = grad + (1-p1) * vel + p1 * vel2
        grad_norm = torch.norm(grad, p=1)
        grad /= grad_norm
        grad += momentum * decay
        momentum = grad
        images_sign = alpha * grad.sign()
        images_sign = images_sign.cpu().detach().numpy()
        vel2 = vel
        for r in range(45000):
            mutation = np.random.uniform(0, 1)
            attack_cor = set[r]
            attack_row = attack_cor[0]
            attack_row = np.array(attack_row)[0]
            attack_col = attack_cor[1]
            attack_col = np.array(attack_col)[0]
            if mutation < 0.05:
                if r == 0:
                    sattack_cor = set[44999]
                    sattack_row = sattack_cor[0]
                    sattack_row = np.array(sattack_row)[0]
                    sattack_col = sattack_cor[1]
                    sattack_col = np.array(sattack_col)[0]
                    img[0:, 0:, sattack_row, sattack_col] = img[0:, 0:, sattack_row, sattack_col] + images_sign[0:, 0:, sattack_row, sattack_col]
                else:
                    sattack_cor = set[r-1]
                    sattack_row = sattack_cor[0]
                    sattack_row = np.array(sattack_row)[0]
                    sattack_col = sattack_cor[1]
                    sattack_col = np.array(sattack_col)[0]
                    img[0:, 0:, sattack_row, sattack_col] = img[0:, 0:, sattack_row, sattack_col] + images_sign[0:, 0:, sattack_row, sattack_col]
                img[0:, 0:, attack_row, attack_col] = img[0:, 0:, attack_row, attack_col]
            else:
                img[0:, 0:, attack_row, attack_col] = img[0:, 0:, attack_row, attack_col] + images_sign[0:, 0:, attack_row, attack_col]

        img[0:, 0, :, :] = np.clip(img[0:, 0, :, :], dim1_min, dim1_max)
        img[0:, 1, :, :] = np.clip(img[0:, 1, :, :], dim2_min, dim2_max)
        img[0:, 2, :, :] = np.clip(img[0:, 2, :, :], dim3_min, dim3_max)
        img = torch.from_numpy(img)
        images = copy.deepcopy(img)

    return images

test_data = MyDataset(txt=root + 'val.txt', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]))

test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.vgg16(pretrained=True).to(device)

model2 = models.vgg13(pretrained=True).to(device)

loss = nn.CrossEntropyLoss()
model.eval()
model2.eval()
two_net_correct = 0
total_count = 0
nos = 0
nos2 = 0
eps = 0.1
for images, labels in test_loader:
    attack_set = []
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    _, pre = torch.max(output.data, 1)
    output2 = model2(images)
    _, pre2 = torch.max(output2.data, 1)
    total_count += 1
    if pre == pre2 == labels:
        two_net_correct += 1
        if two_net_correct <= 500:
            hot_map = saliency_map(images)
            for i in range(45000):
               attack_ponit = np.where(hot_map == np.max(hot_map))
               attack_set.append(attack_ponit)
               attack_ponit_row = attack_ponit[0]
               attack_ponit_row = np.array(attack_ponit_row)[0]
               attack_ponit_col = attack_ponit[1]
               attack_ponit_col = np.array(attack_ponit_col)[0]
               hot_map[attack_ponit_row, attack_ponit_col] = 0
            adv_image = mbvmifgsm_attack(model, loss, images, labels, eps, attack_set)
            adv_image = adv_image.to(device)
            outputs = model(adv_image)
            _, pres = torch.max(outputs.data, 1)
            if pres == labels:
                nos += 1
            succ_acctak= two_net_correct - nos
            outputs2 = model2(adv_image)
            _, pres2 = torch.max(outputs2.data, 1)
            if pres2 == labels:
                nos2 += 1
            succ_acctak2 = two_net_correct - nos2
            if two_net_correct > 0:
                print('Accuracy of white attack: %f %%' % (100 * float(succ_acctak) / two_net_correct))
                print('Accuracy of black attack: %f %%' % (100 * float(succ_acctak2) / two_net_correct))
            if two_net_correct == 500:
                break



























