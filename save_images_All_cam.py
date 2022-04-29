import cv2
import time
import os

path = "navda_dec15"
try : 
    os.mkdir(path)
except:
    pass

# Dict of all cam
all_cam = { "227":'rtsp://admin:123micro@10.92.197.227/profile1',\
            "228":"rtsp://admin:123micro@10.92.197.228/profile1",\
            "229":"rtsp://admin:123micro@10.92.197.229/profile1",\
            "230":'rtsp://admin:123micro@10.92.197.230/profile1',\
            "237":"rtsp://admin:123micro@10.92.197.237/profile1",\
            "238":"rtsp://admin:123micro@10.92.197.238/profile1",\
            "239":"rtsp://admin:123micro@10.92.197.239/profile1",\
            "241":"rtsp://admin:123micro@10.92.197.241/profile1",\
            "243":'rtsp://admin:123micro@10.92.197.243/profile1',\
            "244":"rtsp://admin:123micro@10.92.197.244/profile1",\
            "245":"rtsp://admin:123micro@10.92.197.245/profile1",\
            "248":"rtsp://admin:123micro@10.92.197.248/profile1"}

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

    time.sleep(300)


