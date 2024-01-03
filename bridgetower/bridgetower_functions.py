from transformers import BridgeTowerModel, BridgeTowerProcessor
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
from torch.nn.functional import pad
import math

def msr_inputs(folder_path, captions):
        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")

        # Get images
        images = []
        if os.path.exists(folder_path):
                items = os.listdir(folder_path)

                for item in items:
                        item_path = os.path.join(folder_path, item)
                        images.append(Image.open(str(item_path)).convert('RGB'))

        num_images = len(images)
        reuse_caption = math.ceil(num_images/len(captions))

        full_captions = [item for item in captions for _ in range(reuse_caption)]
        # Make sure it's the same size as images
        texts = full_captions[:len(images)]

        inputs = [processor(image, text, return_tensors='pt') for image, text in zip(images, texts)]

        num_inputs = len(inputs)
        
        return inputs, num_inputs