import os
import cv2
import time
from datetime import datetime
import random
import glob
root_folder = '/home/ninad/Documents/violations_dashboard/vid_online_feb28/sites'
# path = "/home/ninad/Documents/violations_dashboard/vid_online_feb28

for root in  os.listdir(root_folder):
    print(root)
    path = '/home/ninad/Documents/violations_dashboard/vid_online_feb28/sites/'+root
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(
        'test_vid_'+root+'.mp4', fourcc, 5, (1280, 720))

    for data in os.listdir(path):
        # print(data)
        paths = os.path.join(path, data)
        # print(paths)
        image = cv2.imread(os.path.join(path,data))
        resized = cv2.resize(image, (1280,720), interpolation = cv2.INTER_AREA)

        # cv2.imshow("image",image)
        # cv2.waitKey(0)

        for i in range(5):
            out.write(resized)

    # images = cv2.imread("/home/sr/Pictures/176.png")

    # for i in range(150):
    #     out.write(images)


print("sucessful")