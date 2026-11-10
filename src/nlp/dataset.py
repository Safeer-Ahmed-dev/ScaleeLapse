import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ScaleeLapseDataset(Dataset):
    """
    Custom PyTorch Dataset for loading local commercial images and template-formatted 
    captions for the Vision-Encoder-Decoder multimodal pipeline.
    """
    def __init__(self, csv_path, tokenizer, feature_extractor, max_target_length=128, max_samples=None):
        # Loaded with .head(max_samples) for fast local testing. 
        # Remove .head() or set max_samples=None when running your full 2,000-image training run.
        self.df = pd.read_csv(csv_path)
        if max_samples is not None:
            self.df = self.df.head(max_samples)
            
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        caption = row["caption"]

        # 1. Load image and ensure RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {img_path}: {e}")

        # 2. Extract pixel values using ViT feature extractor
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # 3. Tokenize target text (caption + hashtags) using BART tokenizer
        encoded_target = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        
        labels = encoded_target.input_ids.squeeze(0)
        
        # Replace padding token ids with -100 so they are ignored in the loss function
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

if __name__ == "__main__":
    # Quick sanity check / test block updated for BART
    from transformers import AutoTokenizer, AutoImageProcessor
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    
    # Updated from AutoFeatureExtractor to AutoImageProcessor for ViT compatibility
    feature_extractor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    
    dataset = ScaleeLapseDataset("data/raw/metadata.csv", tokenizer, feature_extractor, max_samples=None)
    print(f"Dataset initialized successfully with {len(dataset)} samples!")
    
    sample = dataset[0]
    print("Sample pixel shape:", sample["pixel_values"].shape)
    print("Sample labels shape:", sample["labels"].shape)