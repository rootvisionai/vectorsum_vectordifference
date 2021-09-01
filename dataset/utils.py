from __future__ import print_function
from __future__ import division

import torchvision
from torchvision import transforms
import PIL.Image
import torch
from skimage.filters import unsharp_mask
import numpy as np

def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)

def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class print_shape():
    def __call__(self, im):
        print(im.size)
        return im

class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, `3x8-bit pixels, true color`, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

class pad_shorter():
    def __call__(self, im):
        h,w = im.size[-2:]
        s = max(h, w) 
        new_im = PIL.Image.new("RGB", (s, s))
        new_im.paste(im, ((s-h)//2, (s-w)//2))
        return new_im    

class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul_(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor

class Sharpen(object):
    def __init__(self, radius = 1, amount = 1.5):
        self.radius = radius
        self.amount = amount
    
    def __call__(self, img):
        img = np.array(img)
        out = unsharp_mask(img, radius = self.radius, amount = self.amount, multichannel = True)
        out = PIL.Image.fromarray(np.uint8(out*255)).convert('RGB')
        return out
    
    def __repr__(self):
        return self.__class__.__name__ + '(radius={0}, amount={1})'.\
            format(self.radius, self.amount)

class CustomRotation(object):
    def __init__(self, fill=128,
                 padding_mode=None):
        
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric',None]
        self.fill = fill
        self.padding_mode = padding_mode
    
    def __call__(self, img, angle):
        """
        Args:img (PIL Image): Image to be rotated.
        Returns:PIL Image: Padded(if padding_mode is not None) randomly rotated and centercropped image.
        """
        if self.padding_mode:
            centercrop_size = img.size
            pad_size_r = centercrop_size[1]//2
            pad_size_c = centercrop_size[0]//2
            img = torchvision.transforms.functional.pad(img, (pad_size_r,pad_size_c), fill = self.fill, padding_mode = self.padding_mode)
        else:
            centercrop_size = img.size[0]/2
        img = torchvision.transforms.functional.rotate(img,angle=angle,resample=PIL.Image.BICUBIC)
        img = torchvision.transforms.functional.center_crop(img, centercrop_size)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(fill={0}, padding_mode={1})'.format(self.fill, self.padding_mode)

class CustomFill(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.fill = fill
        self.padding_mode = padding_mode
    
    def calc_padding(self,img):
        shape = img.size
        
        if shape[0]>=shape[1]:
            increment = shape[0]-shape[1]
            pad_size = (0,increment//2,0,increment//2)
        else:
            increment = shape[1]-shape[0]
            pad_size = (increment//2,0,increment//2,0)
        
        return pad_size
    
    def __call__(self, img):
        """
        Args:img (PIL Image): Image to be padded.
        Returns:PIL Image: Padded image.
        """
        return transforms.functional.pad(img, self.calc_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(fill={}, padding_mode={})'.\
            format(self.fill, self.padding_mode)

class CustomRandomRotation(object):
    def __init__(self, fill=128,
                 padding_mode=None,
                 rotation_range = (30,270)):
        
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric',None]
        self.fill = fill
        self.padding_mode = padding_mode
        self.RandomRotation = transforms.RandomRotation(rotation_range, resample=PIL.Image.BICUBIC)
    
    def __call__(self, img):
        """
        Args:img (PIL Image): Image to be rotated.
        Returns:PIL Image: Padded(if padding_mode is not None) randomly rotated and centercropped image.
        """
        if self.padding_mode:
            centercrop_size = img.size
            pad_size_r = centercrop_size[1]//2
            pad_size_c = centercrop_size[0]//2
            img = transforms.functional.pad(img, (pad_size_r,pad_size_c), fill = self.fill, padding_mode = self.padding_mode)
        else:
            centercrop_size = img.size[0]//2, img.size[1]//2
        img = self.RandomRotation(img)
        img = torchvision.transforms.functional.center_crop(img, centercrop_size)
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + '(fill={}, padding_mode={})'.\
            format(self.fill, self.padding_mode)


def make_transform(is_train = True, is_inception = False, angle = 0., resize_overwrite = None):
    # Resolution Resize List : 256, 292, 361, 512
    # Resolution Crop List: 224, 256, 324, 448
    
    resnet_sz_resize = 224
    resnet_sz_crop = 224
    if resize_overwrite:
        resnet_sz_resize = resize_overwrite
        resnet_sz_crop = resize_overwrite
    # resnet_mean = [0.485, 0.456, 0.406]
    # resnet_std = [0.229, 0.224, 0.225]
    
    # For largely cropped images, keep padding_mode = None in CustomRandomRotation
    # For square images, comment CustomFill
    
    resnet_transform = transforms.Compose(
    [
        #CustomFill(padding_mode='constant',fill=128),
        CustomRandomRotation(padding_mode = None, rotation_range = (15,345)) if is_train else Identity(),
        transforms.RandomResizedCrop(size=resnet_sz_crop,scale=(0.8, 1)) if is_train else Identity(),
        #transforms.RandomHorizontalFlip() if True else Identity(),
        transforms.CenterCrop(resnet_sz_crop),
        transforms.Resize(resnet_sz_resize),
        transforms.ToTensor(),
        #transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

    inception_sz_resize = 224
    inception_sz_crop = 224
    if resize_overwrite:
        inception_sz_resize = resize_overwrite
        inception_sz_crop = resize_overwrite
    # inception_mean = [104, 117, 128]
    # inception_std = [1, 1, 1]
    inception_transform = transforms.Compose(
    [
        #CustomFill(padding_mode='constant'),
        CustomRandomRotation(padding_mode = None, rotation_range = (15,345)) if is_train else Identity(),
        transforms.RandomResizedCrop(size=inception_sz_crop,scale=(0.8, 1.2)) if is_train else Identity(),
        #transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.CenterCrop(resnet_sz_crop),
        transforms.Resize(inception_sz_resize),
        transforms.ToTensor(),
        ScaleIntensities([0, 1], [0, 255]),
        #transforms.Normalize(mean=inception_mean, std=inception_std)
    ])
    
    return inception_transform if is_inception else resnet_transform