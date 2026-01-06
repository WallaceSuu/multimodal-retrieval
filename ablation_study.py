"""
Ablation Study Script

This script runs evaluation on all trained models and compares their performance.
"""

import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from evaluate_clip import evaluate_model
from train_clip import CLIPImageEncoder, COCODataset
from train_clip_improved import CLIPImageEncoderImproved
from torch.utils.data import DataLoader
from transformers import CLIPModel
import argparse


def load_model(checkpoint_path, use_improved=False):
    """Load model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if use_improved:
        image_encoder = CLIPImageEncoderImproved(embedding_dim=512)
    else:
        image_encoder = CLIPImageEncoder(embedding_dim=512)
    
    image_encoder.load_state_dict(checkpoint['model_state_dict'])
    image_encoder.to(device)
    
    # Load CLIP text encoder
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
    text_encoder = clip_model.text_model
    text_encoder.to(device)
    text_encoder.eval()
    
    return image_encoder, text_encoder, device


def run_ablation_evaluation(checkpoint_dirs, data_root='coco2014', cache_dir='cache', 
                           subset_size=5000, batch_size=32):
    """
    Evaluate all model configurations and compare.
    
    Args:
        checkpoint_dirs: List of checkpoint directory paths
        data_root: COCO data root
        cache_dir: Cache directory
        subset_size: Evaluate on subset for faster evaluation
        batch_size: Batch size for evaluation
    """
    results = {}
    
    # Load validation dataset
    val_dataset = COCODataset(
        images_dir=Path(data_root) / 'images' / 'val2014',
        cache_file=Path(cache_dir) / 'val_embeddings.pt',
        metadata_file=Path(cache_dir) / 'val_metadata.pt'
    )
    
    # Use subset if specified
    if subset_size:
        from torch.utils.data import Subset
        import random
        indices = random.sample(range(len(val_dataset)), min(subset_size, len(val_dataset)))
        val_dataset = Subset(val_dataset, indices)
        print(f"Using subset of {len(val_dataset)} samples for evaluation")
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_path = checkpoint_dir / 'best_model.pt'
        config_path = checkpoint_dir / 'config.json'
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        
        config_name = checkpoint_dir.name.replace('checkpoints_', '')
        print(f"\n{'='*70}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*70}")
        
        # Load config to determine if improved model
        use_improved = False
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                mods = config.get('modifications', {})
                use_improved = mods.get('use_layer_norm', False) or \
                             mods.get('use_batch_norm', False) or \
                             mods.get('use_dropout', False)
        
        try:
            image_encoder, text_encoder, device = load_model(checkpoint_path, use_improved)
            
            # Evaluate
            metrics, _, _, _ = evaluate_model(image_encoder, text_encoder, val_loader, device, batch_size)
            
            results[config_name] = metrics
            
            print(f"\nResults for {config_name}:")
            print(f"  Recall@1 I2T:  {metrics['recall@1_i2t']:.4f}")
            print(f"  Recall@5 I2T:  {metrics['recall@5_i2t']:.4f}")
            print(f"  Recall@10 I2T: {metrics['recall@10_i2t']:.4f}")
            print(f"  Recall@1 T2I:  {metrics['recall@1_t2i']:.4f}")
            print(f"  Recall@5 T2I:  {metrics['recall@5_t2i']:.4f}")
            print(f"  Recall@10 T2I: {metrics['recall@10_t2i']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {config_name}: {e}")
            results[config_name] = {'error': str(e)}
    
    return results


def plot_ablation_results(results, save_path='ablation_comparison.png'):
    """Plot comparison of all configurations."""
    configs = list(results.keys())
    
    # Filter out errors
    valid_configs = [c for c in configs if 'error' not in results[c]]
    if not valid_configs:
        print("No valid results to plot")
        return
    
    # Extract metrics
    recall1_i2t = [results[c]['recall@1_i2t'] for c in valid_configs]
    recall5_i2t = [results[c]['recall@5_i2t'] for c in valid_configs]
    recall10_i2t = [results[c]['recall@10_i2t'] for c in valid_configs]
    recall1_t2i = [results[c]['recall@1_t2i'] for c in valid_configs]
    recall5_t2i = [results[c]['recall@5_t2i'] for c in valid_configs]
    recall10_t2i = [results[c]['recall@10_t2i'] for c in valid_configs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(valid_configs))
    width = 0.25
    
    # Image-to-Text
    axes[0].bar(x - width, recall1_i2t, width, label='Recall@1', alpha=0.8)
    axes[0].bar(x, recall5_i2t, width, label='Recall@5', alpha=0.8)
    axes[0].bar(x + width, recall10_i2t, width, label='Recall@10', alpha=0.8)
    axes[0].set_xlabel('Configuration', fontsize=12)
    axes[0].set_ylabel('Recall', fontsize=12)
    axes[0].set_title('Image-to-Text Retrieval', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(valid_configs, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Text-to-Image
    axes[1].bar(x - width, recall1_t2i, width, label='Recall@1', alpha=0.8)
    axes[1].bar(x, recall5_t2i, width, label='Recall@5', alpha=0.8)
    axes[1].bar(x + width, recall10_t2i, width, label='Recall@10', alpha=0.8)
    axes[1].set_xlabel('Configuration', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('Text-to-Image Retrieval', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(valid_configs, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nAblation comparison plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run ablation study evaluation')
    parser.add_argument('--checkpoint_dirs', type=str, nargs='+', 
                      default=['checkpoints_baseline', 'checkpoints_augmentation', 
                              'checkpoints_layer_norm', 'checkpoints_batch_norm',
                              'checkpoints_dropout', 'checkpoints_warmup',
                              'checkpoints_augmentation_layer_norm',
                              'checkpoints_augmentation_dropout',
                              'checkpoints_layer_norm_dropout',
                              'checkpoints_all'],
                      help='List of checkpoint directories to evaluate')
    parser.add_argument('--data_root', type=str, default='coco2014', help='COCO data root')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--subset_size', type=int, default=5000, 
                       help='Evaluate on subset for faster evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_ablation_evaluation(
        args.checkpoint_dirs,
        args.data_root,
        args.cache_dir,
        args.subset_size,
        args.batch_size
    )
    
    # Save results
    with open('ablation_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: ablation_evaluation_results.json")
    
    # Plot comparison
    plot_ablation_results(results)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        print("\nBest configurations:")
        for metric in ['recall@1_i2t', 'recall@5_i2t', 'recall@10_i2t',
                      'recall@1_t2i', 'recall@5_t2i', 'recall@10_t2i']:
            best_config = max(valid_results.items(), key=lambda x: x[1][metric])
            print(f"  {metric}: {best_config[0]} ({best_config[1][metric]:.4f})")
    
    print("="*70)


if __name__ == '__main__':
    main()

