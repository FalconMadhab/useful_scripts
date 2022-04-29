from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
import os
from detectron2.config import get_cfg
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

for d in ["train", "val"]:
    register_coco_instances(f"elite_{d}", {}, f"trainval.json", f"Final_data/{d}")

dataset_dicts = DatasetCatalog.get("elite_train")
microcontroller_metadata = MetadataCatalog.get("elite_train")

cfg = get_cfg()
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
predictor = DefaultPredictor(cfg)

im = cv2.imread("/home/ninad/Downloads/hello_world/cam5.jpg")
outputs = predictor(im)
# print(outputs)
v = Visualizer(im[:, :, ::-1],
               metadata=microcontroller_metadata, 
               scale=0.8, 
               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)


v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
plt.imsave(os.path.join(os.path.join(cfg.OUTPUT_DIR), str("res") + '.png'), img)