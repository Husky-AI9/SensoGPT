from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
from skimage import io
from PIL import Image
import numpy as np
import os
import re
class SAM:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        self.detection_model = YOLO("yolov8x-obb.pt")  # load an official model
        self.segment_model = FastSAM("FastSAM-x.pt")  # or FastSAM-x.pt

        self.all_dict = {'plane': 1, 'ship': 2, 'storage tank': 3, 'baseball diamond': 4, 'tennis court': 5,
                         'basketball court': 6, 'ground track field': 7, 'harbor': 8, 'bridge': 9,
                         'large vehicle': 10, 'small vehicle': 11, 'helicopter': 12, 'roundabout': 13,
                         'soccer ball field': 14, 'swimming pool': 15}
    def inference(self, image_path, det_prompt ,updated_image_path):
        image = Image.open(image_path)
        results = self.detection_model(image)  # predict on an image 
        if det_prompt.strip().lower() in [i.strip().lower()  for i in self.all_dict.keys()]:
            counter = 0
            bounding_boxes = []
            if len(results) == 0:
                return f"No result."
            for result in results:
                print("------------------------------------------------------")
                for box in result.obb.xyxy:
                    bounding_boxes.append({
                        "x_min": box[0].item(),
                        "y_min": box[1].item(),
                        "x_max": box[2].item(),
                        "y_max": box[3].item()
                    })
                    scale_factor = 0.7
                    scaled_xmin = int(bounding_boxes[0]['x_min'] + (bounding_boxes[0]['x_max'] - bounding_boxes[0]['x_min']) * (1 - scale_factor) / 2)
                    scaled_ymin = int(bounding_boxes[0]['y_min'] + ( bounding_boxes[0]['y_max'] - bounding_boxes[0]['y_min']) * (1 - scale_factor) / 2)
                    scaled_xmax = int(bounding_boxes[0]['x_min'] + (bounding_boxes[0]['x_max'] - bounding_boxes[0]['x_min']) * (1 + scale_factor) / 2)
                    scaled_ymax = int(bounding_boxes[0]['y_min'] + ( bounding_boxes[0]['y_max'] - bounding_boxes[0]['y_min']) * (1 + scale_factor) / 2)
                    cx = int((scaled_xmin + scaled_xmax) / 2.0)
                    cy = int((scaled_ymin + scaled_ymax) / 2.0)
                    print(cx,cy)
                    if counter == 0:
                        everything_results = self.segment_model(image_path, device="cpu", retina_masks=True,conf=0.2, iou=0.9)
                        prompt_process = FastSAMPrompt(image_path, everything_results, device="cpu")
                    else:
                        everything_results = self.segment_model(updated_image_path, device="cpu", retina_masks=True,conf=0.2, iou=0.9)
                        prompt_process = FastSAMPrompt(updated_image_path, everything_results, device="cpu")
                        os.remove(updated_image_path)  # Remove the file

                    bboxes = [scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
                    #ann = prompt_process.box_prompt(bbox=bboxes)
                    ann = prompt_process.point_prompt(points=[[cx, cy]], pointlabel=[1])
                    prompt_process.plot(annotations=ann, output="./segment")

                    # if counter == 0:
                    #     prompt_process.plot(annotations=ann, output="./segment")

                    # else:
                    #     prompt_process.plot(annotations=ann, output="./segment/1")

                    match = re.search(r'[^\\]+$', image_path)
                    file_name = match.group()
                    
                    updated_image_path = f'segment/{file_name}'
                    bounding_boxes = []
                    counter += 1


            print(f"\nProcessed Instance Segmentation, Input Image: {image_path + ',' + det_prompt}, Output SegMap: {updated_image_path}")
            return updated_image_path
        else:
            print(f"\nCategory: { det_prompt} is not supported. Please use other tools.")
            return f"Category {det_prompt} is not supported. Please use other tools."



