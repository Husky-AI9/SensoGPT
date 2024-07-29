import os

import re
import uuid
from skimage import io
import argparse
import inspect
from langchain.chat_models import ChatOpenAI
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
import numpy as np
from Prefix import  RS_SENSOGPT_PREFIX, RS_SENSOGPT_FORMAT_INSTRUCTIONS, RS_SENSOGPT_SUFFIX
from RStask import ImageEdgeFunction,CaptionFunction,DetectionFunction,LanduseFunction,InstanceFunction
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import GPT4All

GEMINI_API_KEY = ""

class MediaItem(BaseModel):
    data: str
    mimeType: str

class RequestBody(BaseModel):
    message: str
    media: List[str]
    media_types: List[str]
    general_settings: dict
    safety_settings: dict

app = FastAPI()

# Allow CORS for your frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

os.makedirs('image', exist_ok=True)
def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator
def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}.png'.replace('__','_')
    return os.path.join(head, new_file_name)

class InstanceSegmentation:
    def __init__(self, device):
        print("Initializing InstanceSegmentation")
        self.func=InstanceFunction(device)
    @prompts(name="Instance Segmentation for Remote Sensing Image",
             description="useful when you want to apply man-made instance segmentation for the image. The expected input category include plane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, vehicle, helicopter, roundabout, soccer ball field, and swimming pool."
                         "like: extract plane from this image, "
                         "or predict the ship in this image, or extract tennis court from this image, segment harbor from this image, Extract the vehicle in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from plane, or ship, or storage tank, or baseball diamond, or tennis court, or basketball court, or ground track field, or harbor, or bridge, or vehicle, or helicopter, or roundabout, or soccer ball field, or  swimming pool. ")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="instance_" + det_prompt)
        text=self.func.inference(image_path, det_prompt,updated_image_path)
        return text

class LandUseSegmentation:
    def __init__(self, device):
        print("Initializing LandUseSegmentation")
        self.func=LanduseFunction(device)

    @prompts(name="Land Use Segmentation for Remote Sensing Image",
             description="useful when you want to apply land use gegmentation for the image. The expected input category include Building, Road, Water, Barren, Forest, Farmland, Landuse."
                         "like: generate landuse map from this image, "
                         "or predict the landuse on this image, or extract building from this image, segment roads from this image, Extract the water bodies in the image. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text of the category,selected from Lnad Use, or Building, or Road, or Water, or Barren, or Forest, or Farmland, or Landuse.")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="landuse")
        text = det_prompt+' segmentation result in '+updated_image_path
        text=self.func.inference(image_path, det_prompt,updated_image_path)
        return text

class ObjectDetection:
    def __init__(self, device):
        self.func=DetectionFunction(device)


    @prompts(name="Detect the given object",
             description="useful when you only want to detect the bounding box of the certain objects in the picture according to the given text."
                         "like: detect the plane, or can you locate an object for me."
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")

    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        updated_image_path = get_new_image_name(image_path, func_name="detection_" + det_prompt.replace(' ', '_'))
        log_text=self.func.inference(image_path, det_prompt,updated_image_path)
        return log_text

class EdgeDetection:
    def __init__(self, device):
        print("Initializing Edge Detection Function....")
        self.func = ImageEdgeFunction()
    @prompts(name="Edge Detection On Image",
             description="useful when you want to detect the edge of the remote sensing image. "
                         "like: detect the edges of this image, or canny detection on image, "
                         "or perform edge detection on this image, or detect the  edge of this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        updated_image_path=get_new_image_name(inputs, func_name="edge")
        self.func.inference(inputs,updated_image_path)
        return updated_image_path

class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.func=CaptionFunction(device,GEMINI_API_KEY)
    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        captions = self.func.inference(image_path)
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions

class SensoChatGPT:
    def __init__(self, gpt_name,load_dict,proxy_url):
        print(f"Initializing SensoChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for SensoChatGPT")
        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)
        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))

        self.llm = GPT4All(model="Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf")
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def initialize(self):
        self.memory.clear() #clear previous history
        PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = RS_SENSOGPT_PREFIX, RS_SENSOGPT_FORMAT_INSTRUCTIONS, RS_SENSOGPT_SUFFIX
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,stop=["\nObservation:", "\n\tObservation:"],
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,'suffix': SUFFIX}, )

    def run_text(self, text, state):
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state
    def run_image(self, image_dir, state, txt=None):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        img = io.imread(image_dir)
        io.imsave(image_filename, img.astype(np.uint8))
        description = self.models['ImageCaptioning'].inference(image_filename)
        Human_prompt = f' Provide a remote sensing image named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
        AI_prompt = "Received."
        self.memory.chat_memory.add_user_message(Human_prompt)
        self.memory.chat_memory.add_ai_message(AI_prompt)
        #print(f'-----------Human_prompt {Human_prompt}')

        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        state=self.run_text(f'{txt} {image_filename} ', state)
        last_state = len(state)-1
        return state, state[last_state]

proxy_url = None
gpt_name="gpt-4-turbo"
state = []
load = "ImageCaptioning_cuda:0,ObjectDetection_cuda:0,LandUseSegmentation_cuda:0,InstanceSegmentation_cuda:0,EdgeDetection_cpu"
load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in load.split(',')}
bot = SensoChatGPT(gpt_name=gpt_name,load_dict=load_dict,proxy_url=proxy_url)
bot.initialize()
print('SensoChatGPT initialization done, you can now chat with SensoChatGPT~')

@app.post("/run-image")
async def run_image_endpoint(request: RequestBody):
    global state
    # Decode and save the images
    for i, media in enumerate(request.media):
        image_data = base64.b64decode(media)
        image_path = f'image/{uuid.uuid4()}.png'
        with open(image_path, "wb") as buffer:
            buffer.write(image_data)
    
    result_image = request.media
    state, result = bot.run_image(image_path,state,txt=request.message)
    result_string = result[1]
    
    if "land use segmentation" in result_string:
        # Extract the predicted image path from result[0]
      # Regular expression to find substrings that start with 'image' and end with '.png'
        pattern = r'\S*_landuse_\S*'


        # Find all matches
        matches = re.findall(pattern, result_string)

        cleaned_string = matches[0]
        png_position = cleaned_string.find('png') + len('png')
        cleaned_string = cleaned_string[:png_position]
        cleaned_string = cleaned_string.replace("![](file=","")
        cleaned_string = cleaned_string.strip('"')

        # Print the matches
        # Read and encode the predicted image
        with open(cleaned_string, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        result_image = encoded_image
    
    elif "edge detection" in result_string:
        pattern = r'\S*_edge_\S*'
        # Find all matches
        matches = re.findall(pattern, result_string)

        cleaned_string = matches[0]
        png_position = cleaned_string.find('png') + len('png')
        cleaned_string = cleaned_string[:png_position]
        cleaned_string = cleaned_string.replace("![](file=","")
        cleaned_string = cleaned_string.strip('"')
        # Print the matches
        # Read and encode the predicted image
        with open(cleaned_string, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        result_image = encoded_image


    elif "runs/obb/" in result_string:
        # Extract the predicted image path from result[0]
        pattern = r"\S*runs/obb/[^ ]*"
        # Find all matches
        matches = re.findall(pattern, result_string)
        predicted_image_path = matches[0]
        predicted_image_path = predicted_image_path.strip('"')
        print(predicted_image_path)
        png_position = predicted_image_path.find('png') + len('png')
        cleaned_string = predicted_image_path[:png_position]
        cleaned_string = cleaned_string.replace("![](file=","")
        # Read and encode the predicted image
        with open(cleaned_string, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        result_image = encoded_image
    
    elif "segment" in result_string:
        pattern = r"\S*segment/[^ ]*"
        # Find all matches
        matches = re.findall(pattern, result_string)
        predicted_image_path = matches[0]
        predicted_image_path = predicted_image_path.strip('"')
        print(predicted_image_path)
        png_position = predicted_image_path.find('png') + len('png')
        cleaned_string = predicted_image_path[:png_position]
        cleaned_string = cleaned_string.replace("![](file=","")
        # Read and encode the predicted image
        with open(cleaned_string, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        result_image = encoded_image
    
    return JSONResponse(content={"status": "success", "prompt_result": result_string, "image": result_image})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)