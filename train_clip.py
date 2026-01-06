"""
CLIP Fine-tuning Training Script

This script implements:
1. ResNet50 image encoder with ImageNet pretrained weights
2. Projection head (2 linear layers with GELU) mapping to 512-dim CLIP space
3. Frozen CLIP text encoder
4. InfoNCE loss for training
5. Training and validation loops with logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import CLIPModel
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
import json
from datetime import datetime


class CLIPImageEncoder(nn.Module):
    """
    ResNet50 image encoder with projection head.
    
    Architecture:
    - ResNet50 backbone (ImageNet pretrained)
    - Projection head: Linear -> GELU -> Linear -> 512-dim embedding
    """
    
    def __init__(self, embedding_dim=512):
        super(CLIPImageEncoder, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove avgpool and fc
        
        # Get the feature dimension from ResNet50 (2048)
        backbone_dim = 2048
        
        # Projection head: 2 linear layers with GELU activation
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through image encoder.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            Image embeddings [batch_size, embedding_dim]
        """
        # Extract features from ResNet50
        features = self.backbone(x)  # [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        
        # Project to CLIP embedding space
        embeddings = self.projection(features)
        
        # L2 normalize (standard in CLIP)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings


class COCODataset(Dataset):
    """
    Dataset class for COCO images and cached text embeddings.
    """
    
    # CLIP normalization values
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    IMAGE_SIZE = 224
    
    def __init__(self, images_dir, cache_file, metadata_file):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            cache_file: Path to cached embeddings .pt file
            metadata_file: Path to metadata .pt file
        """
        self.images_dir = Path(images_dir)
        
        # Load cached data
        print(f"Loading cached data from {cache_file}")
        cached_data = torch.load(cache_file)
        self.text_embeddings = cached_data['embeddings']
        self.image_ids = cached_data['image_ids']
        self.filenames = cached_data['filenames']
        
        # Load metadata
        metadata = torch.load(metadata_file)
        self.metadata = metadata
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        ])
        
        print(f"Loaded {len(self.image_ids)} samples")
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get image and corresponding text embedding.
        
        Args:
            idx: Sample index
            
        Returns:
            image: Preprocessed image tensor
            text_embedding: Cached text embedding
            filename: Image filename (for debugging)
        """
        filename = self.filenames[idx]
        image_path = self.images_dir / filename
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # If image fails to load, return a black image
            print(f"Warning: Failed to load {image_path}: {e}")
            image = torch.zeros(3, self.IMAGE_SIZE, self.IMAGE_SIZE)
        
        # Get cached text embedding
        text_embedding = self.text_embeddings[idx]
        
        return image, text_embedding, filename


def info_nce_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Compute InfoNCE (Contrastive) loss for CLIP.
    
    InfoNCE loss maximizes similarity between positive pairs (image, caption)
    and minimizes similarity between negative pairs.
    
    Args:
        image_embeddings: Image embeddings [batch_size, embedding_dim]
        text_embeddings: Text embeddings [batch_size, embedding_dim]
        temperature: Temperature scaling parameter
        
    Returns:
        loss: InfoNCE loss scalar
    """
    batch_size = image_embeddings.size(0)
    
    # Compute similarity matrix
    # image_embeddings: [batch_size, dim]
    # text_embeddings: [batch_size, dim]
    # similarity: [batch_size, batch_size]
    similarity = torch.matmul(image_embeddings, text_embeddings.t()) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=image_embeddings.device)
    
    # Cross-entropy loss in both directions (symmetric)
    loss_i2t = nn.functional.cross_entropy(similarity, labels)
    loss_t2i = nn.functional.cross_entropy(similarity.t(), labels)
    
    # Average the two directions
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss


class CLIPTrainer:
    """
    Trainer class for fine-tuning CLIP.
    """
    
    def __init__(self, 
                 image_encoder,
                 text_encoder,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-4,
                 temperature=0.07):
        """
        Initialize trainer.
        
        Args:
            image_encoder: ResNet50 image encoder with projection head
            text_encoder: Frozen CLIP text encoder
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate for optimizer
            temperature: Temperature for InfoNCE loss
        """
        self.image_encoder = image_encoder.to(device)
        self.text_encoder = text_encoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.temperature = temperature
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Only image encoder and projection are trainable
        trainable_params = sum(p.numel() for p in self.image_encoder.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer (only for image encoder)
        self.optimizer = optim.AdamW(
            self.image_encoder.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10,  # 10 epochs
            eta_min=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_times = []
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.image_encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, text_embeddings, _) in enumerate(pbar):
            images = images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            image_embeddings = self.image_encoder(images)
            
            # Compute InfoNCE loss
            loss = info_nce_loss(image_embeddings, text_embeddings, self.temperature)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.image_encoder.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for images, text_embeddings, _ in pbar:
                images = images.to(self.device)
                text_embeddings = text_embeddings.to(self.device)
                
                # Forward pass
                image_embeddings = self.image_encoder(images)
                
                # Compute InfoNCE loss
                loss = info_nce_loss(image_embeddings, text_embeddings, self.temperature)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=10, save_dir='checkpoints'):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("Starting Training")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            self.train_times.append(epoch_time)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}\n")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.image_encoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_dir / 'best_model.pt')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})\n")
        
        total_time = time.time() - start_time
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_times': self.train_times,
            'total_time': total_time,
            'num_epochs': num_epochs,
            'device': str(self.device)
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"Average time per epoch: {np.mean(self.train_times):.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {save_dir}")
        print(f"{'='*70}\n")
        
        return history
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(12, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Time per epoch
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_times, 'g-', label='Time per Epoch', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.close()


def main():
    """Main training function."""
    # Configuration
    config = {
        'data_root': 'coco2014',
        'cache_dir': 'cache',
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'temperature': 0.07,
        'save_dir': 'checkpoints',
        'clip_model': 'openai/clip-vit-base-patch32'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Load CLIP text encoder (frozen)
    print(f"\nLoading CLIP text encoder: {config['clip_model']}")
    clip_model = CLIPModel.from_pretrained(config['clip_model'], use_safetensors=True)
    text_encoder = clip_model.text_model
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    print("✓ Text encoder loaded and frozen")
    
    # Create image encoder
    print("\nCreating ResNet50 image encoder with projection head...")
    image_encoder = CLIPImageEncoder(embedding_dim=512)
    print("✓ Image encoder created")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = COCODataset(
        images_dir=Path(config['data_root']) / 'images' / 'train2014',
        cache_file=Path(config['cache_dir']) / 'train_embeddings.pt',
        metadata_file=Path(config['cache_dir']) / 'train_metadata.pt'
    )
    
    val_dataset = COCODataset(
        images_dir=Path(config['data_root']) / 'images' / 'val2014',
        cache_file=Path(config['cache_dir']) / 'val_embeddings.pt',
        metadata_file=Path(config['cache_dir']) / 'val_metadata.pt'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create trainer
    trainer = CLIPTrainer(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        temperature=config['temperature']
    )
    
    # Train
    history = trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    # Plot training curves
    trainer.plot_training_curves(save_path=Path(config['save_dir']) / 'training_curves.png')
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Hardware: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total training time: {history['total_time']/60:.2f} minutes")
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")
    print(f"Final val loss: {history['val_losses'][-1]:.4f}")
    print(f"Best val loss: {min(history['val_losses']):.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

