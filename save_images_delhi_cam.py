import cv2
import time
import os

path = "delhi_jan28"
try : 
    os.mkdir(path)
except:
    pass

# Dict of all cam
all_cam = { "401":'rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/401',\
            "69":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/501",\
            "70":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/901",\
            "74":'rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/1001',\
            "76":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/1101",\
            "78":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/1201",\
            "64":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/1401",\
            "79":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/301",\
            "101":"rtsp://admin:123456%40a@192.168.1.42:554/streaming/channels/101"}

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


