import matplotlib.pyplot as plt
import json
import os
import numpy as np
from PIL import Image
from datasets import DatasetDict, Dataset
from src.utilities import *

class memecap_dataset:
    def __init__(self, target_width, target_height):
        self.target_width = target_width
        self.target_height = target_height
        # Paths to JSON files
        self.test_json_path = 'data/meme-cap/data/memes-test.json'
        self.train_val_json_path = 'data/meme-cap/data/memes-trainval.json'
        
        # Load the JSON data
        self.test_text_data = self.load_json(self.test_json_path)
        self.trainval_text_data = self.load_json(self.train_val_json_path)

    def load_json(self, file_path):
        """Load JSON file."""
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    
    def load_datasets(self, from_idx = 0, to_idx = -1):
        """Load images and return a DatasetDict."""
        datasets = {}
        directory = 'data/memes'
        data_splits = {
            'test': self.test_text_data,
            'trainval': self.trainval_text_data
        }
        for split, data in data_splits.items():
            images = []
            captions = []
            for item in data[from_idx:to_idx]:
                img_path = os.path.join(directory, item['img_fname'])
                
                try:
                    with Image.open(img_path) as img:
                        
                        if self.target_width > 0 and self.target_height > 0:
                            img = resize_and_crop_image(img, self.target_width, self.target_height)
                        img_array = np.array(img)
                        images.append(img_array)
                        captions.append(item.get('meme_captions', [""])[0])
                except IOError:
                    print(f"Error opening image {img_path}")
            datasets[split] = Dataset.from_dict({
                'image': images, # image elements will be transformed into lists
                'caption': captions
            })
        
        self.dataset = DatasetDict(datasets)
    
    def load_images(self, image_fnames):
        """Load images and return a list of Images."""
        images = []
        directory = 'data/memes'
        for item in image_fnames:
            img_path = os.path.join(directory, item['img_fname'])
            try:
                with Image.open(img_path) as img:
                    img_array = np.array(img)
                    images.append(img_array)
            except IOError:
                print(f"Error opening image {img_path}")
        return images

# # Usage example:
# meme_loader = memecap_dataset()
# meme_loader.load_datasets(0, 10)
# print(meme_loader.dataset)
# visualize_meme(meme_loader.dataset['test']['image'][0], meme_loader.dataset['test']['caption'][0])
