"""
Improved CLIP Fine-tuning Training Script

This script includes modifications to improve accuracy:
1. Data augmentation (random crop, color jitter, horizontal flip)
2. Layer normalization in projection head
3. Dropout for regularization
4. Learning rate warmup
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

from train_clip import CLIPTrainer, info_nce_loss, COCODataset


class CLIPImageEncoderImproved(nn.Module):
    """
    Improved ResNet50 image encoder with:
    - Layer normalization in projection head
    - Batch normalization option
    - Dropout for regularization
    """
    
    def __init__(self, embedding_dim=512, use_layer_norm=True, use_batch_norm=False, dropout=0.1):
        super(CLIPImageEncoderImproved, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        backbone_dim = 2048
        
        # Improved projection head with normalization options and dropout
        layers = []
        layers.append(nn.Linear(backbone_dim, embedding_dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(embedding_dim))
        
        layers.append(nn.GELU())
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(embedding_dim, embedding_dim))
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        elif use_layer_norm:
            layers.append(nn.LayerNorm(embedding_dim))
        
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.projection(features)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


class COCODatasetAugmented(COCODataset):
    """
    COCO dataset with data augmentation for training.
    """
    
    def __init__(self, images_dir, cache_file, metadata_file, augment=False):
        super().__init__(images_dir, cache_file, metadata_file)
        self.augment = augment
        
        if augment:
            # Training augmentations
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(self.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
            ])
        else:
            # Validation: no augmentation
            self.transform = transforms.Compose([
                transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
            ])


class ImprovedCLIPTrainer(CLIPTrainer):
    """
    Improved trainer with learning rate warmup.
    """
    
    def __init__(self, *args, warmup_epochs=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * len(self.train_loader)
        self.current_step = 0
        
        # Override scheduler with warmup
        self.base_lr = kwargs.get('learning_rate', 1e-4)
        self.scheduler = None  # Will be set in train_epoch
    
    def get_lr(self, epoch, step):
        """Get learning rate with warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_progress = (epoch * len(self.train_loader) + step) / self.warmup_steps
            return self.base_lr * warmup_progress
        else:
            # Cosine annealing after warmup
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            return self.base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    def train_epoch(self, epoch):
        """Train for one epoch with warmup."""
        self.image_encoder.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (images, text_embeddings, _) in enumerate(pbar):
            images = images.to(self.device)
            text_embeddings = text_embeddings.to(self.device)
            
            # Update learning rate with warmup
            if epoch < self.warmup_epochs or self.scheduler is None:
                lr = self.get_lr(epoch, batch_idx)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                self.scheduler.step()
            
            # Forward pass
            self.optimizer.zero_grad()
            image_embeddings = self.image_encoder(images)
            
            # Compute InfoNCE loss
            loss = info_nce_loss(image_embeddings, text_embeddings, self.temperature)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.image_encoder.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs=10, save_dir='checkpoints'):
        """Override to set num_epochs for warmup."""
        self.num_epochs = num_epochs
        return super().train(num_epochs, save_dir)


def train_with_modifications(config_name='baseline', modifications=None):
    """
    Train model with specified modifications.
    
    Args:
        config_name: Name of configuration (for saving)
        modifications: Dict with modification flags:
            - use_augmentation: bool
            - use_layer_norm: bool
            - use_batch_norm: bool
            - use_dropout: bool
            - dropout_rate: float
            - use_warmup: bool
            - warmup_epochs: int
    """
    if modifications is None:
        modifications = {}
    
    # Default modifications
    use_augmentation = modifications.get('use_augmentation', False)
    use_layer_norm = modifications.get('use_layer_norm', False)
    use_batch_norm = modifications.get('use_batch_norm', False)
    use_dropout = modifications.get('use_dropout', False)
    dropout_rate = modifications.get('dropout_rate', 0.1)
    use_warmup = modifications.get('use_warmup', False)
    warmup_epochs = modifications.get('warmup_epochs', 2)
    
    # Configuration
    config = {
        'data_root': 'coco2014',
        'cache_dir': 'cache',
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'temperature': 0.07,
        'save_dir': f'checkpoints_{config_name}',
        'clip_model': 'openai/clip-vit-base-patch32'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training Configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Modifications:")
    print(f"  - Data Augmentation: {use_augmentation}")
    print(f"  - Layer Normalization: {use_layer_norm}")
    print(f"  - Batch Normalization: {use_batch_norm}")
    print(f"  - Dropout: {use_dropout} (rate: {dropout_rate})")
    print(f"  - Learning Rate Warmup: {use_warmup} ({warmup_epochs} epochs)")
    print(f"{'='*70}\n")
    
    # Load CLIP text encoder (use cached if available)
    print(f"Loading CLIP text encoder: {config['clip_model']}")
    try:
        # Try to load from cache first (offline mode if network fails)
        clip_model = CLIPModel.from_pretrained(
            config['clip_model'], 
            use_safetensors=True,
            local_files_only=False  # Allow download if not cached
        )
    except Exception as e:
        print(f"Warning: Could not load from HuggingFace ({e})")
        print("Attempting to load from local cache only...")
        try:
            clip_model = CLIPModel.from_pretrained(
                config['clip_model'], 
                use_safetensors=True,
                local_files_only=True  # Only use cache
            )
        except Exception as e2:
            print(f"Error: Model not found in cache. Please ensure you have internet connection")
            print(f"or that the model was previously downloaded. Error: {e2}")
            raise
    
    text_encoder = clip_model.text_model
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Create image encoder with modifications
    print("Creating improved image encoder...")
    image_encoder = CLIPImageEncoderImproved(
        embedding_dim=512,
        use_layer_norm=use_layer_norm,
        use_batch_norm=use_batch_norm,
        dropout=dropout_rate if use_dropout else 0.0
    )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = COCODatasetAugmented(
        images_dir=Path(config['data_root']) / 'images' / 'train2014',
        cache_file=Path(config['cache_dir']) / 'train_embeddings.pt',
        metadata_file=Path(config['cache_dir']) / 'train_metadata.pt',
        augment=use_augmentation
    )
    
    val_dataset = COCODatasetAugmented(
        images_dir=Path(config['data_root']) / 'images' / 'val2014',
        cache_file=Path(config['cache_dir']) / 'val_embeddings.pt',
        metadata_file=Path(config['cache_dir']) / 'val_metadata.pt',
        augment=False  # No augmentation for validation
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
    if use_warmup:
        trainer = ImprovedCLIPTrainer(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config['learning_rate'],
            temperature=config['temperature'],
            warmup_epochs=warmup_epochs
        )
    else:
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
    
    # Save configuration
    config_path = Path(config['save_dir']) / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'config_name': config_name,
            'modifications': modifications,
            **config
        }, f, indent=2)
    
    # Plot training curves
    trainer.plot_training_curves(save_path=Path(config['save_dir']) / 'training_curves.png')
    
    return history, trainer


def main():
    """Run ablation study with different modifications."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CLIP with modifications')
    parser.add_argument('--config', type=str, default='baseline',
                      choices=['baseline', 'augmentation', 'layer_norm', 'batch_norm', 
                              'dropout', 'warmup', 'augmentation_layer_norm', 
                              'augmentation_dropout', 'layer_norm_dropout', 'all', 'ablation'],
                      help='Configuration to train')
    args = parser.parse_args()
    
    # Define configurations
    configs = {
        'baseline': {
            'use_augmentation': False,
            'use_layer_norm': False,
            'use_batch_norm': False,
            'use_dropout': False,
            'use_warmup': False
        },
        'augmentation': {
            'use_augmentation': True,
            'use_layer_norm': False,
            'use_batch_norm': False,
            'use_dropout': False,
            'use_warmup': False
        },
        'layer_norm': {
            'use_augmentation': False,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'use_dropout': False,
            'use_warmup': False
        },
        'batch_norm': {
            'use_augmentation': False,
            'use_layer_norm': False,
            'use_batch_norm': True,
            'use_dropout': False,
            'use_warmup': False
        },
        'dropout': {
            'use_augmentation': False,
            'use_layer_norm': False,
            'use_batch_norm': False,
            'use_dropout': True,
            'dropout_rate': 0.1,
            'use_warmup': False
        },
        'warmup': {
            'use_augmentation': False,
            'use_layer_norm': False,
            'use_batch_norm': False,
            'use_dropout': False,
            'use_warmup': True,
            'warmup_epochs': 2
        },
        'augmentation_layer_norm': {
            'use_augmentation': True,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'use_dropout': False,
            'use_warmup': False
        },
        'augmentation_dropout': {
            'use_augmentation': True,
            'use_layer_norm': False,
            'use_batch_norm': False,
            'use_dropout': True,
            'dropout_rate': 0.1,
            'use_warmup': False
        },
        'layer_norm_dropout': {
            'use_augmentation': False,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'use_dropout': True,
            'dropout_rate': 0.1,
            'use_warmup': False
        },
        'all': {
            'use_augmentation': True,
            'use_layer_norm': True,
            'use_batch_norm': False,
            'use_dropout': True,
            'dropout_rate': 0.1,
            'use_warmup': True,
            'warmup_epochs': 2
        }
    }
    
    if args.config == 'ablation':
        # Run configurations in specific order:
        # 1. Individual modifications first
        # 2. Then combined modifications
        
        print("Running full ablation study...")
        print("\nTraining order:")
        print("1. Individual modifications (augmentation, layer_norm, batch_norm, dropout, warmup)")
        print("2. Combined modifications (all)")
        print("="*70)
        
        # Define training order
        individual_configs = ['augmentation', 'layer_norm', 'batch_norm', 'dropout', 'warmup']
        combined_configs = ['all']
        
        results = {}
        
        # Step 1: Train individual modifications
        print("\n" + "="*70)
        print("STEP 1: Training Individual Modifications")
        print("="*70)
        
        for config_name in individual_configs:
            if config_name not in configs:
                continue
                
            mods = configs[config_name]
            checkpoint_dir = Path(f'checkpoints_{config_name}')
            checkpoint_file = checkpoint_dir / 'best_model.pt'
            
            # Skip if already trained
            if checkpoint_file.exists():
                print(f"\n{'='*70}")
                print(f"Skipping {config_name} - already trained")
                print(f"Checkpoint exists: {checkpoint_file}")
                print(f"{'='*70}\n")
                # Load existing results if available
                history_file = checkpoint_dir / 'training_history.json'
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    results[config_name] = {
                        'final_train_loss': history['train_losses'][-1],
                        'final_val_loss': history['val_losses'][-1],
                        'best_val_loss': min(history['val_losses']),
                        'total_time': history['total_time']
                    }
                continue
            
            print(f"\n\n{'='*70}")
            print(f"Training Individual Modification: {config_name}")
            print(f"{'='*70}\n")
            try:
                history, trainer = train_with_modifications(config_name, mods)
                results[config_name] = {
                    'final_train_loss': history['train_losses'][-1],
                    'final_val_loss': history['val_losses'][-1],
                    'best_val_loss': min(history['val_losses']),
                    'total_time': history['total_time']
                }
            except Exception as e:
                print(f"Error training {config_name}: {e}")
                import traceback
                traceback.print_exc()
                results[config_name] = {'error': str(e)}
        
        # Step 2: Train combined modifications
        print("\n" + "="*70)
        print("STEP 2: Training Combined Modifications")
        print("="*70)
        
        for config_name in combined_configs:
            if config_name not in configs:
                continue
                
            mods = configs[config_name]
            checkpoint_dir = Path(f'checkpoints_{config_name}')
            checkpoint_file = checkpoint_dir / 'best_model.pt'
            
            # Skip if already trained
            if checkpoint_file.exists():
                print(f"\n{'='*70}")
                print(f"Skipping {config_name} - already trained")
                print(f"Checkpoint exists: {checkpoint_file}")
                print(f"{'='*70}\n")
                # Load existing results if available
                history_file = checkpoint_dir / 'training_history.json'
                if history_file.exists():
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    results[config_name] = {
                        'final_train_loss': history['train_losses'][-1],
                        'final_val_loss': history['val_losses'][-1],
                        'best_val_loss': min(history['val_losses']),
                        'total_time': history['total_time']
                    }
                continue
            
            print(f"\n\n{'='*70}")
            print(f"Training Combined Modifications: {config_name}")
            print(f"{'='*70}\n")
            try:
                history, trainer = train_with_modifications(config_name, mods)
                results[config_name] = {
                    'final_train_loss': history['train_losses'][-1],
                    'final_val_loss': history['val_losses'][-1],
                    'best_val_loss': min(history['val_losses']),
                    'total_time': history['total_time']
                }
            except Exception as e:
                print(f"Error training {config_name}: {e}")
                import traceback
                traceback.print_exc()
                results[config_name] = {'error': str(e)}
        
        # Save ablation results
        with open('ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n\n" + "="*70)
        print("Ablation study complete! Results saved to ablation_results.json")
        print("="*70)
    else:
        # Train single configuration
        mods = configs.get(args.config, configs['baseline'])
        train_with_modifications(args.config, mods)


if __name__ == '__main__':
    main()

