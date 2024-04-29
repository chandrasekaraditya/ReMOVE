import os
import time
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from crop import find_smallest_bounding_square, draw_bb

# FileNotFoundError: [Errno 2] No such file or directory: 'sam_vit_h_4b8939.pth'
# Download the model in root dir from URL - https://huggingface.co/spaces/abhishek/StableSAM/blob/main/sam_vit_h_4b8939.pth

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").cuda()
predictor = SamPredictor(sam)
DATA_DIR = 'examples/sdinpaint'
MASK_DIR = 'examples/masks'
bname = '823000000001'          # Bad Inpainting
# bname = '122000000009'          # Good Inpainting
crop = True

if crop:
    binary_image = np.array(Image.open(os.path.join(MASK_DIR, bname+'.jpg')))
    # draw_bb(os.path.join(MASK_DIR, bname+'.jpg'))
    x, y, size = find_smallest_bounding_square(binary_image)
    
    input = np.array(Image.open(os.path.join(DATA_DIR, bname+'.jpg')).convert("RGB"))[y:y+size, x:x+size]
    
    mask_fg = np.array(Image.fromarray(np.array(Image.open(os.path.join(MASK_DIR, bname+'.jpg')).convert("L"))[y:y+size, x:x+size]).resize((64,64))).reshape((1,1,64,64))//255
    mask_bg = 1 - mask_fg
    
else:
    input = np.array(Image.open(os.path.join(DATA_DIR, bname+'.jpg')))
    
    mask_fg = np.array(Image.open(os.path.join(MASK_DIR,bname+'.jpg')).resize((64,64))).reshape((1,1,64,64))//255
    mask_bg = 1 - mask_fg
    
embeddings = predictor.get_aggregate_features(input, [mask_fg, mask_bg])

remove_score = cosine_similarity(embeddings[0], embeddings[1]).item()

print("The ReMOVE Score for Quality of Object Erasure is:", remove_score)

