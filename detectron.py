import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

from detectron2.data.datasets import register_coco_instances

for d in ["train", "val"]:
    register_coco_instances(f"elite_{d}", {}, f"trainval.json", f"Final_data/{d}")

import random
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

dataset_dicts = DatasetCatalog.get("elite_train")
microcontroller_metadata = MetadataCatalog.get("elite_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("elite_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00020
cfg.SOLVER.MAX_ITER = 80000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.OUTPUT_DIR = "/home/ninad/detectron/output2"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=True)
# trainer.train()
print("Training Completed!")

#Multiple_image_inferencing
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor

# cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0039999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
predictor = DefaultPredictor(cfg)
root_folder = '/home/ninad/detectron/test_data/'
for root in  os.listdir(root_folder):
    print(root)
    path = '/home/ninad/detectron/test_data/'+ root
    im = cv2.imread(path)
    outputs = predictor(im)
    # print(outputs)
    v = Visualizer(im[:, :, ::-1],
                metadata=microcontroller_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )


    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
    plt.imsave("test_res_next/"+ root + '.png', img)

'''
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor

# cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
predictor = DefaultPredictor(cfg)

im = cv2.imread("/home/ninad/detectron/test_data/test_1.jpg")
outputs = predictor(im)
print(outputs)
v = Visualizer(im[:, :, ::-1],
               metadata=microcontroller_metadata, 
               scale=0.8, 
               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)


v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR)+"test_res/res100" + '.png'), img)
'''