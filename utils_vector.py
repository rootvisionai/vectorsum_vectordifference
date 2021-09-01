import numpy as np
import torch
import torch.nn.functional as F
import os
from skimage import io
import shutil

def save_proxies(cfg, filename, proxies, label_map):
    try:
        os.mkdir('../proxies/model_{}'.format(cfg.dataset))
    except:
        pass
    try:
        os.mkdir('{}'.format('../proxies'+cfg.resume))
    except:
        pass
    data = {'proxies': proxies, 'label_map': label_map}
    torch.save(data,'../proxies'+cfg.resume+'/{}.pth'.format(filename))

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    # print('T.shape',T.shape,'Y.shape',T.shape)
    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader, device):
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch_id, batch in enumerate(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = J.to(device)
                    J = model(J) #.cuda())

                for j in J:
                    A[i].append(j)
    
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A)) if i!=2]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader, nearest_neighbours, device):
    # calculate embeddings with model and get targets
    
    X, T = predict_batchwise(model, dataloader, device)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = nearest_neighbours
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        # print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

# def generate_candidate_proxies(dl_cand):
    

def save_debug_images(cfg, models_dir, dl, mode, range_ = 1):
    try:
        shutil.rmtree('{}/{}'.format(cfg.debug_images,models_dir.split('/')[-1]))
    except:
        pass
    try:
        os.mkdir(cfg.debug_images)
    except:
        pass
    try:
        os.mkdir('{}/{}'.format(cfg.debug_images,models_dir.split('/')[-1]))
        # print('{}/{}'.format(cfg.debug_images,models_dir.split('/')[-1]))
    except:
        pass
    try:
        os.mkdir('{}/{}/{}'.format(cfg.debug_images,models_dir.split('/')[-1],mode))
        # print('{}/{}/{}'.format(cfg.debug_images,models_dir.split('/')[-1],mode))
    except:
        pass
    for batch_idx, (x, y, y_str) in enumerate(dl):
        if batch_idx>range_:
            break
        for idx in range(10):
            io.imsave('{}/{}/{}/batchid@{}_id@{}_label@{}.png'.format(cfg.debug_images,
                      models_dir.split('/')[-1],
                      mode,batch_idx,
                      idx,y_str[idx]),
                      (x[idx].permute(1,2,0).numpy()*255).astype(np.uint8))