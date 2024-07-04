from ultralytics import YOLO

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import os
from skimage import io
from PIL import Image
class YOLOOBB:
    def __init__(self, device):
        self.model = YOLO("yolov8x-obb.pt")  # load an official model


    def inference(self, image_path, det_prompt,updated_image_path):
        image = Image.open(image_path)
        results = self.model(image,save = True,show_labels=True, save_txt=False)  # predict on an image
        print('---------------------------------------------')
        for result in results:
            print(result.save_dir)
            # List all files in the directory
            files = os.listdir(result.save_dir)

            # Filter to find the image file (assuming it has a common image file extension)
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
            image_files = [f for f in files if f.endswith(image_extensions)]

           
            image_path_new = os.path.join(result.save_dir, image_files[0])
            print(image_path_new)

            return det_prompt+' object detection result in '+ image_path_new


