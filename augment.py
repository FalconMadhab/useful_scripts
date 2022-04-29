import argparse
from ast import arg
import os
import cv2
import copy
from cv2 import add
from torchvision import transforms
import numpy as np
import imutils
from PIL import Image
import random

#run the command "python3 augment.py" along with these tags:::
# -i for input
# -d for output
# -p for percentage of data to perform augmentation on
# -r wether you want to randomly apply the augmentation
# -a the list of augmentations like this "123456" without spaces each number indicates the augmentation


#1
def rrotate(image,target,filename):
    rota_image_30 = imutils.rotate(image,random.randint(1,20))
    #rota_image_n30 = imutils.rotate(image,random.randint(1,-30))
    cv2.imwrite(target+'/rotated30_'+filename,rota_image_30)
    #cv2.imwrite(target+'/rotatedn30_'+filename,rota_image_n30)

#2
def invertt(image,target,filename):
    inve = np.invert(image)
    cv2.imwrite(target+'/inverted_'+filename,inve)

#3
def gray_scale(image,target,filename):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(target+'/grayscale_'+filename,gray)

#4
def increase_brightness(img,dest_folder,i,brightness=430,contrast=300):

    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,img, 0, ga_mma)
        cv2.imwrite(dest_folder+'/brightness_'+i,cal)

#5
def fliptheimg(img_loc,dest,i):
    original_img = Image.open(img_loc)
        
    # Flip the original image horizontally
    horz_img = original_img.transpose(method=Image.FLIP_LEFT_RIGHT)
    horz_img.save(dest+'/flip_h'+i)
    vert_img= original_img.transpose(method=Image.FLIP_TOP_BOTTOM)
    vert_img.save(dest+'/flip_v'+i)

#6
def addblur(img_loc,dest,i):
    img = Image.open(img_loc)
    transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5))
    ])
    imgt = transform(img)
    imgt.save(dest+'/blurr_'+i)
    img.close()

#7
def cutmix(img_loc,dest,filename,p=0.010):
    imgg = Image.open(img_loc)
    img_arr = np.array(imgg)
    width, height = imgg. size
    # Turning the pixel values of the 400x400 pixels to black
    numh = random.randint(0,int(height/3)+int(height/2))
    numw = random.randint(0,int(width/3)+int(width/2))
    img_arr[numh : int(height*p)+numh,numw : int(width*p)+numw] = (0, 0, 0)

    # Creating an image out of the previously modified array
    imgg = Image.fromarray(img_arr)
    imgg.save(dest+'/cutmix'+filename)
    imgg.verify()
    imgg.close()

#8
def percent_crop(img_loc,des,filename,p=0.15):
    im = Image.open(img_loc)

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = im.size

    # Setting the points for cropped image
    left = 0
    top = 0
    right = width
    bottom = height

    # Cropped image of above dimension
    # (It will not change original image)
    
    a = random.randint(1,4)
    if a== 1:
        im1 = im.crop((left, top, right*(1-p), bottom))
    elif a == 2:
        im1 = im.crop((left, top, right, bottom*(1-p)))
    elif a == 3:
        im1 = im.crop((width*p, top, right, bottom))
    elif a == 4:
        im1 = im.crop((left,height*p,right,bottom))
    
    
    im1.save(des+'/crop__'+filename)
    im.close()

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def auto_aug(aug_num,img,dest,filename,img_loc):
    if aug_num == 1:
        rrotate(img,dest,filename)
    elif aug_num == 2:
        invertt(img,dest,filename)
    elif aug_num == 3:
        gray_scale(img,dest,filename)
    elif aug_num == 4:
        increase_brightness(img,dest,filename,brightness=random.randint(70,400))
    elif aug_num == 5:
        fliptheimg(img_loc,dest,filename)
    elif aug_num == 6:
        addblur(img_loc,dest,filename)
    elif aug_num == 7:
        cutmix(img_loc,dest,filename,random.randint(20,25)/100)
    elif aug_num == 8:
        percent_crop(img_loc,dest,filename,random.randint(15,20)/100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, default = 70,help='percentage of the data to apply the augmentations to')
    parser.add_argument('-i', type=dir_path, default = 70,help='image source location',required=True)
    parser.add_argument('-r',choices=[1,0], type=int, default = 1,help='if you want apply the augmentations randomly or not')
    parser.add_argument('-a', type=str, default = '1234567',help='1--- rotate\n2--- invert\n3--- grayscale\n4--- increasebrightness\n5--- percentcrop\n6--- fliptheimage\n7--- blur')
    parser.add_argument('-d',type=str,default=os.getcwd()+'/augmented_images',help='specify the detsination')
    args = parser.parse_args()
    image_list = os.listdir(args.i)
    source = args.i
    aug_list = []
    for j in args.a:
        aug_list.append(int(j))
    if not os.path.exists(args.d):
        os.makedirs(args.d)
    len_of_img = len(image_list)
    for i in range(len_of_img):
        if i < len_of_img*(args.p)/100:
            img_loc = source+'/'+image_list[i]
            img=cv2.imread(img_loc)
            if args.r == 1:
                num = random.randint(0,len(aug_list)-1)
                aug = aug_list[num]
                auto_aug(aug,img,args.d,image_list[i],img_loc)

            elif args.r == 0:
                for k in aug_list:
                    auto_aug(k,img,args.d,image_list[i],img_loc)
        else:
            break

if __name__ == '__main__':
    main()
'''
img_loc = '/home/shalom/taskss/fsm_classifier/jagad/fsm/fsm_wtf/HPCL_BKC_7-SMOKING_ON_RO-2022-02-23_11-14-520.jpg'
dest='/home/shalom/taskss/fsm_classifier/jagad/fsm/test_aug'

for i in range(20):
    cutmix(img_loc,dest,str(i)+'cut36.jpg',p=0.20)
'''