"""
MS COCO 2014 Dataset Preparation for CLIP-based Training

This script prepares the COCO 2014 dataset by:
1. Loading images and captions
2. Preprocessing images (resize to 224x224, CLIP normalization)
3. Encoding captions using CLIP text encoder
4. Caching text embeddings for faster training
5. Verifying dataset integrity
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from pathlib import Path


class COCODatasetPreparator:
    """
    Prepares COCO 2014 dataset for CLIP-based training.
    """
    
    # CLIP normalization values
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    
    # Image size for CLIP
    IMAGE_SIZE = 224
    
    def __init__(self, 
                 data_root='coco2014',
                 train_images_dir='images/train2014',
                 val_images_dir='images/val2014',
                 train_captions_file='annotations/captions_train2014.json',
                 val_captions_file='annotations/captions_val2014.json',
                 cache_dir='cache',
                 model_name='openai/clip-vit-base-patch32'):
        """
        Initialize the dataset preparator.
        
        Args:
            data_root: Root directory of COCO dataset
            train_images_dir: Relative path to training images
            val_images_dir: Relative path to validation images
            train_captions_file: Relative path to training captions JSON
            val_captions_file: Relative path to validation captions JSON
            cache_dir: Directory to save cached embeddings
            model_name: HuggingFace model name for CLIP
        """
        self.data_root = Path(data_root)
        self.train_images_dir = self.data_root / train_images_dir
        self.val_images_dir = self.data_root / val_images_dir
        self.train_captions_file = self.data_root / train_captions_file
        self.val_captions_file = self.data_root / val_captions_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP model and processor
        print(f"Loading CLIP model: {model_name}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # Prefer safetensors format to avoid torch version issues
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        ])
        
        # Text preprocessing info
        self.tokenizer = self.processor.tokenizer
        self.max_length = self.tokenizer.model_max_length  # Usually 77 for CLIP
        
    def load_captions(self, captions_file):
        """
        Load captions from COCO JSON file.
        
        Args:
            captions_file: Path to captions JSON file
            
        Returns:
            Dictionary mapping image_id to list of captions
        """
        if not captions_file.exists():
            print("\n" + "="*70)
            print("ERROR: Caption file not found!")
            print("="*70)
            print(f"Missing file: {captions_file}")
            print("\nTo fix this:")
            print("1. Run: python download_captions.py")
            print("2. Or download manually from:")
            print("   - Official: https://cocodataset.org/#download")
            print("   - Kaggle: https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3")
            print("\nPlace the files in: coco2014/annotations/")
            print("="*70)
            raise FileNotFoundError(
                f"Caption file not found: {captions_file}\n"
                f"Run 'python download_captions.py' for download instructions."
            )
        
        print(f"Loading captions from {captions_file}")
        with open(captions_file, 'r') as f:
            data = json.load(f)
        
        # Create mapping: image_id -> list of captions
        image_id_to_captions = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            if image_id not in image_id_to_captions:
                image_id_to_captions[image_id] = []
            image_id_to_captions[image_id].append(caption)
        
        # Also create image_id to filename mapping
        image_id_to_filename = {}
        for img_info in data['images']:
            image_id_to_filename[img_info['id']] = img_info['file_name']
        
        return image_id_to_captions, image_id_to_filename
    
    def encode_captions(self, captions, batch_size=32):
        """
        Encode captions using CLIP text encoder.
        
        Args:
            captions: List of caption strings
            batch_size: Batch size for encoding
            
        Returns:
            Tensor of text embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
                batch_captions = captions[i:i+batch_size]
                
                # Tokenize and encode
                inputs = self.processor(
                    text=batch_captions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
                
                embeddings.append(text_features.cpu())
        
        return torch.cat(embeddings, dim=0)
    
    def prepare_split(self, split='train'):
        """
        Prepare a dataset split (train or val).
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Dictionary with prepared data and statistics
        """
        print(f"\n{'='*60}")
        print(f"Preparing {split.upper()} split")
        print(f"{'='*60}")
        
        # Determine paths based on split
        if split == 'train':
            images_dir = self.train_images_dir
            captions_file = self.train_captions_file
            cache_file = self.cache_dir / 'train_embeddings.pt'
            cache_metadata_file = self.cache_dir / 'train_metadata.pt'
        else:
            images_dir = self.val_images_dir
            captions_file = self.val_captions_file
            cache_file = self.cache_dir / 'val_embeddings.pt'
            cache_metadata_file = self.cache_dir / 'val_metadata.pt'
        
        # Load captions
        image_id_to_captions, image_id_to_filename = self.load_captions(captions_file)
        
        # Check if cache exists
        if cache_file.exists() and cache_metadata_file.exists():
            print(f"Loading cached embeddings from {cache_file}")
            cached_data = torch.load(cache_file)
            metadata = torch.load(cache_metadata_file)
            return {
                'embeddings': cached_data['embeddings'],
                'image_ids': cached_data['image_ids'],
                'captions': cached_data['captions'],
                'filenames': cached_data['filenames'],
                'metadata': metadata
            }
        
        # Prepare data
        print("Collecting image-caption pairs...")
        image_ids = []
        all_captions = []
        filenames = []
        
        for image_id, captions_list in tqdm(image_id_to_captions.items(), desc="Collecting pairs"):
            filename = image_id_to_filename.get(image_id)
            if filename and (images_dir / filename).exists():
                # Use first caption for each image (can be modified to use all)
                image_ids.append(image_id)
                all_captions.append(captions_list[0])  # Using first caption
                filenames.append(filename)
        
        print(f"Found {len(image_ids)} valid image-caption pairs")
        
        # Encode captions
        print("Encoding captions with CLIP text encoder...")
        text_embeddings = self.encode_captions(all_captions)
        
        # Prepare metadata
        metadata = {
            'split': split,
            'num_samples': len(image_ids),
            'image_size': self.IMAGE_SIZE,
            'normalization_mean': self.CLIP_MEAN,
            'normalization_std': self.CLIP_STD,
            'tokenizer': 'CLIP (HuggingFace)',
            'max_length': self.max_length,
            'padding_strategy': 'max_length',
            'truncation': True,
            'embedding_dim': text_embeddings.shape[1]
        }
        
        # Cache embeddings
        print(f"Caching embeddings to {cache_file}")
        torch.save({
            'embeddings': text_embeddings,
            'image_ids': image_ids,
            'captions': all_captions,
            'filenames': filenames
        }, cache_file)
        torch.save(metadata, cache_metadata_file)
        
        return {
            'embeddings': text_embeddings,
            'image_ids': image_ids,
            'captions': all_captions,
            'filenames': filenames,
            'metadata': metadata
        }
    
    def load_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        return self.image_transform(image)
    
    def verify_dataset(self, train_data, val_data, num_samples=5):
        """
        Verify dataset integrity by displaying random image-caption pairs.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            num_samples: Number of random samples to display
        """
        print(f"\n{'='*60}")
        print(f"Verifying Dataset Integrity")
        print(f"{'='*60}")
        
        # Create reverse transform for visualization
        reverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-m/s for m, s in zip(self.CLIP_MEAN, self.CLIP_STD)],
                std=[1/s for s in self.CLIP_STD]
            ),
            transforms.ToPILImage()
        ])
        
        # Sample from training set
        print(f"\nDisplaying {num_samples} random samples from TRAINING set:")
        train_indices = random.sample(range(len(train_data['image_ids'])), num_samples)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for idx, sample_idx in enumerate(train_indices):
            image_id = train_data['image_ids'][sample_idx]
            filename = train_data['filenames'][sample_idx]
            caption = train_data['captions'][sample_idx]
            
            # Load image
            image_path = self.train_images_dir / filename
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                image_resized = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
                
                axes[idx].imshow(image_resized)
                axes[idx].set_title(f"Image ID: {image_id}\nCaption: {caption}", fontsize=10)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f"Image not found: {filename}", 
                              ha='center', va='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('train_samples_verification.png', dpi=150, bbox_inches='tight')
        print("Saved training samples to 'train_samples_verification.png'")
        plt.close()
        
        # Sample from validation set
        print(f"\nDisplaying {num_samples} random samples from VALIDATION set:")
        val_indices = random.sample(range(len(val_data['image_ids'])), 
                                   min(num_samples, len(val_data['image_ids'])))
        
        fig, axes = plt.subplots(len(val_indices), 1, figsize=(12, 3*len(val_indices)))
        if len(val_indices) == 1:
            axes = [axes]
        
        for idx, sample_idx in enumerate(val_indices):
            image_id = val_data['image_ids'][sample_idx]
            filename = val_data['filenames'][sample_idx]
            caption = val_data['captions'][sample_idx]
            
            # Load image
            image_path = self.val_images_dir / filename
            if image_path.exists():
                image = Image.open(image_path).convert('RGB')
                image_resized = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
                
                axes[idx].imshow(image_resized)
                axes[idx].set_title(f"Image ID: {image_id}\nCaption: {caption}", fontsize=10)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f"Image not found: {filename}", 
                              ha='center', va='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('val_samples_verification.png', dpi=150, bbox_inches='tight')
        print("Saved validation samples to 'val_samples_verification.png'")
        plt.close()
        
        print("\n✓ Dataset verification complete!")
    
    def print_statistics(self, train_data, val_data):
        """
        Print dataset statistics and preprocessing information.
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
        """
        print(f"\n{'='*60}")
        print("DATASET STATISTICS AND PREPROCESSING INFORMATION")
        print(f"{'='*60}")
        
        print("\n📊 Dataset Split Sizes:")
        print(f"  Training samples: {train_data['metadata']['num_samples']:,}")
        print(f"  Validation samples: {val_data['metadata']['num_samples']:,}")
        print(f"  Total samples: {train_data['metadata']['num_samples'] + val_data['metadata']['num_samples']:,}")
        
        print("\n🖼️  Image Preprocessing:")
        print(f"  Resize: → {train_data['metadata']['image_size']} × {train_data['metadata']['image_size']}")
        print(f"  Normalization Mean: {train_data['metadata']['normalization_mean']}")
        print(f"  Normalization Std: {train_data['metadata']['normalization_std']}")
        
        print("\n📝 Text Preprocessing:")
        print(f"  Tokenizer: {train_data['metadata']['tokenizer']}")
        print(f"  Max Length: {train_data['metadata']['max_length']}")
        print(f"  Padding Strategy: {train_data['metadata']['padding_strategy']}")
        print(f"  Truncation: {train_data['metadata']['truncation']}")
        print(f"  Text Embedding Dimension: {train_data['metadata']['embedding_dim']}")
        
        print("\n💾 Caching:")
        print(f"  Train embeddings cached: {self.cache_dir / 'train_embeddings.pt'}")
        print(f"  Val embeddings cached: {self.cache_dir / 'val_embeddings.pt'}")
        
        print(f"\n{'='*60}\n")


def main():
    """
    Main function to prepare the COCO dataset.
    """
    # Initialize preparator
    preparator = COCODatasetPreparator()
    
    # Prepare training split
    train_data = preparator.prepare_split('train')
    
    # Prepare validation split
    val_data = preparator.prepare_split('val')
    
    # Print statistics
    preparator.print_statistics(train_data, val_data)
    
    # Verify dataset integrity
    preparator.verify_dataset(train_data, val_data, num_samples=5)
    
    print("✅ Dataset preparation complete!")
    print(f"\nCached embeddings are available in: {preparator.cache_dir}/")
    print("You can now use these cached embeddings for training.")


if __name__ == '__main__':
    main()

