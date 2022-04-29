from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.io import read_image
import argparse
import os
import re

def crop_mask(img_path,mask_path,dest):
    fname= img_path.split('/')[-1]
    transform = transforms.Resize((720,1280))
    ppp = transforms.ToPILImage()
    plt.rcParams["savefig.bbox"] = "tight"
    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    img = transform(read_image(img_path))
    mask = read_image(mask_path)

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask == obj_ids[:, None, None]
    boxes = masks_to_boxes(masks)
    try:
        b = boxes.numpy()[0]
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2])
        y2 = int(b[3])
        vvv = F.crop(img,y1,x1,y2-y1,x2-x1)
        fff = ppp(vvv)
        fff.save(dest+'/crop_from_mask_'+fname)
    except:
        print(f'there is nothing masked for this image {img_path} with the mask {mask_path}')
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=dir_path,help='input images bro or sis hastag woke af')
    parser.add_argument('-o',type=dir_path,help='output here macha')
    parser.add_argument('-m',type=dir_path,help = 'the mask images bro or sis u know the rest ')
    args = parser.parse_args()
    input_img = args.i
    destination = args.o
    mask_loc = args.m
    slist = sorted_alphanumeric([x for x in os.listdir(input_img)])
    mlist = sorted_alphanumeric([y for y in os.listdir(mask_loc)])
    for i in range(len(slist)):
        crop_mask(input_img+'/'+slist[i],mask_loc+'/'+mlist[i],destination)

if __name__=='__main__':
    main()