import os 
import cv2
import pandas as pd

path = "/home/sr/Downloads/Violations.xlsx"

file = pd.read_excel(path)

cam_name = file["Source"]
site_id = file["SiteId"]
url = file["BlobURL"]

# site_id[{cam_name--> violations }]

print(cam_name)
print(site_id)
print(url)

dictionary = dict()

for siteid in site_id:
    dictionary[siteid] = dict()
    
