import os
import time
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor
from crop import find_smallest_bounding_square, draw_bb
import argparse

import warnings
warnings.filterwarnings("ignore")

def get_score(predictor, args):
    if args.crop:
        binary_image = np.array(Image.open(args.mask_path))
        if args.draw:
            draw_bb(args.mask_path)
        x, y, size = find_smallest_bounding_square(binary_image)
        
        input_img = np.array(Image.open(args.image_path).convert("RGB"))[y:y+size, x:x+size]
        
        mask_fg = np.array(Image.fromarray(np.array(Image.open(args.mask_path).convert("L"))[y:y+size, x:x+size]).resize((64,64))).reshape((1,1,64,64))//255
        mask_bg = 1 - mask_fg
        
    else:
        input_img = np.array(Image.open(args.image_path).convert("RGB"))
        
        mask_fg = np.array(Image.open(args.mask_path).resize((64,64))).reshape((1,1,64,64))//255
        mask_bg = 1 - mask_fg
        
    embeddings = predictor.get_aggregate_features(input_img, [mask_fg, mask_bg])

    remove_score = cosine_similarity(embeddings[0], embeddings[1]).item()

    return remove_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="arguments for ReMOVE")
    parser.add_argument("-i" , "--image_path", type=str, help="Path to the image file.")
    parser.add_argument("-m", "--mask_path", type=str, help="Path to the corresponding mask file.")
    parser.add_argument("--crop", action="store_true", default=False, help="Crop the image using a bounding box around the mask.")
    parser.add_argument("--draw", action="store_true", default=False, help="Draws a bounding box around the object if crop is chosen.")

    args = parser.parse_args()

    sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").cuda()
    predictor = SamPredictor(sam)

    remove_score = get_score(predictor, args)
    print("The ReMOVE Score for Quality of Object Erasure is:", remove_score)
