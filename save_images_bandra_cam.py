import cv2
import time
import os

path = "navda_dec15"
try : 
    os.mkdir(path)
except:
    pass

# Dict of all cam
all_cam = { "68":'rtsp://admin:AS%2F23441@192.168.0.68/live',\
            "69":"rtsp://admin:AS%2F23441@192.168.0.69/live",\
            "70":"rtsp://admin:AS%2F23441@192.168.0.70/live",\
            "74":'rtsp://admin:AS%2F23441@192.168.0.74/live',\
            "76":"rtsp://admin:AS%2F23441@192.168.0.76/live",\
            "78":"rtsp://admin:AS%2F23441@192.168.0.78/live",\
            "64":"rtsp://admin:AS%2F23441@192.168.0.64/live",\
            "79":"rtsp://admin:AS%2F23441@192.168.0.79/live"}

# all_cam = {
#             "DOCK_15_INSIDE_CAM1":"rtsp://admin:spoton%40123@10.109.8.18/streaming/channels/701",\
#             "ENTRY_CAM":"rtsp://admin:spoton%40123@10.109.8.19/streaming/channels/1001",\
#             "EXIT_CAM":"rtsp://admin:spoton%40123@10.109.8.19/streaming/channels/201"}

while True:

    for i in all_cam.keys():
        print(i)
        print(all_cam[i])
        cap = cv2.VideoCapture(all_cam[i])
        
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame,(1920,1080))
            cv2.imwrite("{}/{}_{}_.jpg".format(path,i,str(time.time())), frame)

    time.sleep(10)


