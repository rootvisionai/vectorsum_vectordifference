import torch
from torchvision import transforms
import itertools
import numpy as np
from config import cfg

out_size = cfg.input_size
rct = transforms.Compose([transforms.ToPILImage(),
                          transforms.Resize((out_size,out_size)),
                          transforms.ToTensor()])
def random_crop(image, crop_size, num_of_crops):
    _,img_row,img_col = image.shape
    crops, bboxes = [], []
    crops.append(rct(image))
    for i in range(num_of_crops):
        coefficients = np.random.rand(2)
        top_left = np.array([coefficients[0]*img_row/2, coefficients[1]*img_col/2],dtype=int)
        bottom_right = top_left+crop_size
        # print("top left row:",top_left[0],"bottom right row:",bottom_right[0],
        #       "top left col:",top_left[1],"bottom right col:",bottom_right[1])
        crop = image[:,top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        crops.append(rct(crop))
        bboxes.append(np.array([top_left, bottom_right]))
    image_out = torch.stack(crops,dim=0)
    bbox_out  = np.array(bboxes)
    return image_out, bbox_out

def create_combinations(bboxes):
    return list(itertools.combinations(np.arange(0, len(bboxes)), 2))

def calc_iou(bbox1, bbox2, orig_img_size = 224):
    template1 = np.zeros((orig_img_size,orig_img_size))
    template2 = np.zeros((orig_img_size,orig_img_size))

    template1[bbox1[0][0]:bbox1[1][0], bbox1[0][1]:bbox1[1][1]] = 1
    template2[bbox2[0][0]:bbox2[1][0], bbox2[0][1]:bbox2[1][1]] = 1

    iou_mask = template1+template2
    _,cnts = np.unique(iou_mask, return_counts=True)
    if len(cnts) == 3:
        iou = cnts[2]/(cnts[1]+cnts[2])
    else:
        iou = 0
    return torch.tensor([iou])

def calculate_ious(combinations, bboxes):
    ious = []
    for comb in combinations:
        ious.append(calc_iou(bboxes[comb[0]], bboxes[comb[1]]))
    return torch.stack(ious, dim=0)

def calculate_ious_batch(combinations, bboxes):
    ious = []
    for bbox_elm in bboxes:
        ious.append(calculate_ious(combinations, bbox_elm))
    return torch.stack(ious, dim=0)

def calculate_ious_for_img(bboxes, orig_img_size = 224):
    template = np.zeros((orig_img_size,orig_img_size))
    for bbox in bboxes:
        template[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]] = 1
    
    _,cnts = np.unique(template, return_counts=True)
    if len(cnts) == 2:
        iou = cnts[1]/(cnts[0]+cnts[1])
    else:
        iou = 0
    return torch.tensor([iou])

def calculate_ious_for_img_batch(bboxes):
    ious = []
    for bbox_elm in bboxes:
        ious.append(calculate_ious_for_img(bbox_elm))
    return torch.stack(ious, dim=0)

def normalize_vector(x):
    norm = x.norm(p=2, dim=1, keepdim=True)
    x_normalized = x.div(norm.expand_as(x))
    return x_normalized

def calculate_cosdists(combinations, emb_vectors):
    emb_vectors = normalize_vector(emb_vectors)
    sim_mat = torch.nn.functional.linear(emb_vectors, emb_vectors)
    similarity_vector = torch.zeros(1,len(combinations)).squeeze(0)
    for cnt, comb in enumerate(combinations):
        similarity_vector[cnt] = sim_mat[comb[0], comb[1]]
    del sim_mat
    return similarity_vector

def calculate_cosdists_batch(combinations_batch, emb_vectors):
    similarity_vector_batch = []
    for i,combinations in enumerate(combinations_batch):
        similarity_vector_batch.append(calculate_cosdists(combinations,
                                                          emb_vectors[i]))
    
    return torch.stack(similarity_vector_batch, dim=0) 

# image = torch.rand((3,512,512))
# crop_size = 512
# num_of_crops = 4
# image_out, bbox_out = random_crop(image, crop_size, num_of_crops)
#
# combinations = create_combinations(bbox_out)
#
# ious = calculate_ious(combinations, bbox_out)
#
# bbox_out_batch = np.stack((bbox_out,bbox_out), axis=0)
# ious_batch = calculate_ious_batch(combinations, bbox_out_batch)
#
# emb_vectors = torch.rand((3,512))
# cos_dists = calculate_cosdists(combinations, emb_vectors)