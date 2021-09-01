import torch
import os
import tqdm
import time

from utils import *
import utils_vector as ut_v
from config import cfg

from net.resnet import Resnet18,Resnet34,Resnet50,Resnet101
from net.googlenet import googlenet
from net.bn_inception import bn_inception

import dataset
import dataset.utils as dataset_utils

number_of_crops = 7
crop_size = 224

image_indexes = [i for i in range(0,cfg.batch_size*(number_of_crops+1),(number_of_crops+1))]
crop_indexes = [k for k in range(cfg.batch_size*(number_of_crops+1)) if k not in image_indexes]

os.chdir('./datasets/')
data_root = os.getcwd()

# Build train set
dataset_train = dataset.load(name = dataset,
                             root = data_root,
                             dpath = '/'+cfg.dataset,
                             mode = 'train',
                             transform = dataset_utils.make_transform(
                                 is_train = True,
                                 is_inception = (cfg.model == 'resnet'),
                                 resize_overwrite = cfg.input_size))

dl_train = torch.utils.data.DataLoader(dataset_train,
                                       batch_size = cfg.batch_size,
                                       shuffle = True,
                                       num_workers = cfg.num_workers,
                                       drop_last = True,
                                       pin_memory = True)

# Build test set
dataset_test = dataset.load(name = cfg.dataset,
                            root = data_root,
                            dpath = '/'+cfg.dataset,
                            mode = 'test',
                            transform = dataset.utils.make_transform(
                                is_train = False, 
                                is_inception = (cfg.model == 'bn_inception'),
                                resize_overwrite = cfg.input_size))

dl_test = torch.utils.data.DataLoader(dataset_test,
                                      batch_size = cfg.batch_size,
                                      shuffle = False,
                                      num_workers = cfg.num_workers,
                                      pin_memory = True)

# Build model
model_embedding_size = cfg.embedding_size
if cfg.model.find('googlenet')+1:
    model = googlenet(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet18')+1:
    model = Resnet18(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet34')+1:
    model = Resnet34(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet50')+1:
    model = Resnet50(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet101')+1:
    model = Resnet101(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)

param_groups = [
                {'params': model.parameters()}
               ]

def criterion(to_be, as_is):
    diff = as_is-to_be
    res = torch.where(diff>0,
                torch.abs((torch.exp(4*(diff))-1)).sum(),
                torch.abs((torch.exp(4*(-diff))-1)).sum())
    return res.sum()

# Optimizer Setting
if cfg.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay, momentum = 0.9, nesterov=True)
elif cfg.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay)
elif cfg.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(cfg.lr), alpha=0.9, weight_decay = cfg.weight_decay, momentum = 0.9)
elif cfg.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_decay_step, gamma = cfg.lr_decay_gamma)

if os.path.isfile('../model_last.pth'):
    print('=> loading checkpoint:\n{}'.format('../model_last.pth'))
    checkpoint = torch.load('../model_last.pth', torch.device(cfg.device))
    model.load_state_dict(checkpoint['model_state_dict'])

model.to(cfg.device)
model.train()

for epoch in range(cfg.epochs):
    
    model.eval()
    Recalls = ut_v.evaluate_cos(model, dl_test, 8, cfg.device)
    torch.save({'model_state_dict':model.state_dict()}, '../model_last.pth')
    with open('../{}-logs_last_results.txt'.format(time.time()), 'w') as f:
        f.write('Last Epoch: {}\n'.format(epoch))
        for i in range(4):
            f.write("Last Recall@{}: {:.4f}\n".format(10**i, Recalls[i] * 100))
    model.train()
    
    pbar = tqdm.tqdm(enumerate(dl_train))
    for i, (images, bboxxes, target_int, target_str) in pbar:
        images = images.reshape(images.shape[0]*images.shape[1],
                        images.shape[2],images.shape[3],
                        images.shape[4])
        # for _ in range(4):
        combinations = create_combinations(bboxxes[0])
        combinations_batch = [combinations for i in range(cfg.batch_size)]
        
        ious_batch_for_crops = calculate_ious_batch(combinations, bboxxes)
        ious_batch_for_crops = ious_batch_for_crops.reshape(ious_batch_for_crops.shape[0]*ious_batch_for_crops.shape[1],
                                                            ious_batch_for_crops.shape[2]).permute(1,0)
        ious_batch_for_img = calculate_ious_for_img_batch(bboxxes)
        
        emb_vectors = model(images.to(cfg.device))
        del images
        emb_vectors_crops = emb_vectors[crop_indexes]
        emb_vectors_crops = emb_vectors_crops.reshape(cfg.batch_size,
                                                      int(emb_vectors_crops.shape[0]/cfg.batch_size),
                                                      emb_vectors_crops.shape[1])
        
        emb_vectors_images = emb_vectors[image_indexes]
        emb_vectors_images = emb_vectors_images.reshape(cfg.batch_size,
                                                      int(emb_vectors_images.shape[0]/cfg.batch_size),
                                                      emb_vectors_images.shape[1])
        
        emb_vectors_sum = emb_vectors_crops.sum(dim=1).unsqueeze(1)
        cos_dists_crops = calculate_cosdists_batch(combinations_batch, emb_vectors_crops)
        cos_dists_img_vs_cropsum = calculate_cosdists_batch([[(0,1)] for k in range(cfg.batch_size)],
                                                            torch.cat((emb_vectors_images,emb_vectors_sum),dim=1))

        loss_1 = criterion(cos_dists_crops.flatten().unsqueeze(0).float(),
                           ious_batch_for_crops.float())
        loss_2 = criterion(cos_dists_img_vs_cropsum.float(),
                           ious_batch_for_img.float())
        loss = loss_1 + loss_2
        
        opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        opt.step()
                    
        pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss1: {:.6f} Loss2: {:.6f} LossT: {:.6f}'.format(
                epoch, i+1, len(dl_train),
                100. * (i+1) / len(dl_train),
                loss_1.item(),
                loss_2.item(),
                loss.item()))
        
    # scheduler.step()