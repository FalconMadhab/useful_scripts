from PIL import Image
import os
import shutil
source = '/home/shalom/taskss/fsm_classifier/fms_aug_final_data/fsm_classifier_final_data_28_uptodate/FSM'
dest = '/home/shalom/taskss/fsm_classifier/fms_aug_final_data/fsm_classifier_final_data_28_uptodate/grayscale'

def is_grey_scale(img_path):
    img = Image.open(img_path)
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i,j))
            if r != g != b: 
                return False
    return True
for i in os.listdir(source):
    if is_grey_scale(source+'/'+i):
        shutil.move(source+'/'+i,dest)
