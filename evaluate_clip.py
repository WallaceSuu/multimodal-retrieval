"""
CLIP Model Evaluation Script

This script evaluates the fine-tuned CLIP model by computing:
1. Recall@1, Recall@5, Recall@10 for Image-to-Text retrieval
2. Recall@1, Recall@5, Recall@10 for Text-to-Image retrieval
3. Visualization of retrieval examples
4. Image classification with class labels
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import random

from train_clip import CLIPImageEncoder, COCODataset


def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    Compute Recall@K metrics.
    
    Args:
        similarity_matrix: [N, N] cosine similarity matrix
                          similarity_matrix[i, j] = similarity between image i and text j
        k_values: List of K values to compute recall for
        
    Returns:
        Dictionary with recall metrics for each K
    """
    N = similarity_matrix.size(0)
    device = similarity_matrix.device
    
    # For each image, find the rank of its correct caption
    # Correct pairs are on the diagonal
    ranks_i2t = []
    ranks_t2i = []
    
    # Image-to-Text: for each image, find rank of its caption
    for i in range(N):
        # Get similarities for image i with all captions
        similarities = similarity_matrix[i, :]
        # Sort in descending order
        sorted_indices = torch.argsort(similarities, descending=True)
        # Find rank of correct caption (should be at index i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks_i2t.append(rank)
    
    # Text-to-Image: for each caption, find rank of its image
    for j in range(N):
        # Get similarities for caption j with all images
        similarities = similarity_matrix[:, j]
        # Sort in descending order
        sorted_indices = torch.argsort(similarities, descending=True)
        # Find rank of correct image (should be at index j)
        rank = (sorted_indices == j).nonzero(as_tuple=True)[0].item() + 1
        ranks_t2i.append(rank)
    
    ranks_i2t = torch.tensor(ranks_i2t, device=device)
    ranks_t2i = torch.tensor(ranks_t2i, device=device)
    
    # Compute Recall@K
    results = {}
    for k in k_values:
        recall_i2t = (ranks_i2t <= k).float().mean().item()
        recall_t2i = (ranks_t2i <= k).float().mean().item()
        results[f'recall@{k}_i2t'] = recall_i2t
        results[f'recall@{k}_t2i'] = recall_t2i
    
    # Also return mean ranks
    results['mean_rank_i2t'] = ranks_i2t.float().mean().item()
    results['mean_rank_t2i'] = ranks_t2i.float().mean().item()
    
    return results


def evaluate_model(image_encoder, text_encoder, data_loader, device, batch_size=32):
    """
    Evaluate model on validation set.
    
    Args:
        image_encoder: Trained image encoder
        text_encoder: CLIP text encoder (frozen)
        data_loader: DataLoader for validation set
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with evaluation metrics
    """
    image_encoder.eval()
    text_encoder.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    print("Computing embeddings...")
    with torch.no_grad():
        for images, text_embeddings, _ in tqdm(data_loader, desc="Processing"):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)
            
            # Get image embeddings
            image_emb = image_encoder(images)
            all_image_embeddings.append(image_emb.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    print(f"Computed embeddings for {len(all_image_embeddings)} samples")
    
    # Compute similarity matrix in batches to save memory
    print("Computing similarity matrix...")
    N = len(all_image_embeddings)
    similarity_matrix = torch.zeros(N, N)
    
    # Process in chunks
    chunk_size = 1000
    for i in tqdm(range(0, N, chunk_size), desc="Computing similarities"):
        end_i = min(i + chunk_size, N)
        for j in range(0, N, chunk_size):
            end_j = min(j + chunk_size, N)
            
            img_emb = all_image_embeddings[i:end_i].to(device)
            txt_emb = all_text_embeddings[j:end_j].to(device)
            
            # Compute cosine similarity (already normalized, so dot product)
            sim = torch.matmul(img_emb, txt_emb.t())
            similarity_matrix[i:end_i, j:end_j] = sim.cpu()
    
    # Compute recall metrics
    print("Computing recall metrics...")
    metrics = compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10])
    
    return metrics, similarity_matrix, all_image_embeddings, all_text_embeddings


def visualize_text_to_image_retrieval(text_query, image_encoder, clip_model, processor, 
                                     dataset, device, top_k=5, save_path='text_to_image_retrieval.png'):
    """
    Visualize top-K image retrievals for a text query.
    
    Args:
        text_query: Text string (e.g., 'sport')
        image_encoder: Trained image encoder
        clip_model: Full CLIP model (for get_text_features)
        processor: CLIP processor for text
        dataset: Dataset to search in
        device: Device to run on
        top_k: Number of top images to retrieve
        save_path: Path to save visualization
    """
    image_encoder.eval()
    clip_model.eval()
    
    # Encode text query using CLIP's get_text_features
    with torch.no_grad():
        inputs = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True).to(device)
        query_embedding = clip_model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    
    # Compute similarities with all images
    print(f"Searching for: '{text_query}'")
    similarities = []
    image_paths = []
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for images, _, filenames in tqdm(data_loader, desc="Computing similarities"):
            images = images.to(device)
            image_embeddings = image_encoder(images)
            
            # Compute similarity
            sim = torch.matmul(query_embedding, image_embeddings.t())
            similarities.append(sim.cpu())
            
            # Store filenames
            image_paths.extend([dataset.images_dir / f for f in filenames])
    
    similarities = torch.cat(similarities, dim=1).squeeze(0)
    
    # Get top-K indices
    top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    # Visualize
    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
    if top_k == 1:
        axes = [axes]
    
    for idx, ax in enumerate(axes):
        img_idx = top_k_indices[idx].item()
        img_path = image_paths[img_idx]
        similarity = similarities[img_idx].item()
        
        # Load and display image
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.set_title(f'Rank {idx+1}\nSim: {similarity:.3f}', fontsize=10)
        except:
            ax.text(0.5, 0.5, f'Image not found\n{img_path.name}', 
                   ha='center', va='center', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Top-{top_k} Image Retrievals for: "{text_query}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    plt.close()


def visualize_image_classification(image_path, image_encoder, clip_model, processor,
                                  class_labels, device, save_path='image_classification.png'):
    """
    Classify an image given a list of class labels.
    
    Args:
        image_path: Path to input image
        image_encoder: Trained image encoder
        clip_model: Full CLIP model (for get_text_features)
        processor: CLIP processor
        class_labels: List of class strings (e.g., ['a person', 'an animal', 'a landscape'])
        device: Device to run on
        save_path: Path to save visualization
    """
    image_encoder.eval()
    clip_model.eval()
    
    # Load and preprocess image
    from train_clip import COCODataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=COCODataset.CLIP_MEAN, std=COCODataset.CLIP_STD)
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Encode image
    with torch.no_grad():
        image_embedding = image_encoder(img_tensor)
    
    # Encode class labels using CLIP's get_text_features
    with torch.no_grad():
        inputs = processor(text=class_labels, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeddings = clip_model.get_text_features(**inputs)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    # Compute similarities
    similarities = torch.matmul(image_embedding, text_embeddings.t()).squeeze(0).cpu().numpy()
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_similarities = similarities[sorted_indices]
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show classification results
    axes[1].barh(range(len(class_labels)), sorted_similarities, color='steelblue')
    axes[1].set_yticks(range(len(class_labels)))
    axes[1].set_yticklabels(sorted_labels, fontsize=11)
    axes[1].set_xlabel('Similarity Score', fontsize=11)
    axes[1].set_title('Classification Scores', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add similarity values on bars
    for i, (label, sim) in enumerate(zip(sorted_labels, sorted_similarities)):
        axes[1].text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved classification to: {save_path}")
    plt.close()
    
    # Print results
    print(f"\nClassification results for: {image_path.name}")
    for i, (label, sim) in enumerate(zip(sorted_labels, sorted_similarities)):
        print(f"  {i+1}. {label}: {sim:.4f}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CLIP model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='coco2014', help='COCO data root')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--subset_size', type=int, default=None, help='Evaluate on subset (for faster eval)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (auto-generated from checkpoint if not specified)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Auto-generate output directory from checkpoint path if not specified
    if args.output_dir is None:
        checkpoint_path = Path(args.checkpoint)
        # Extract config name from checkpoint path (e.g., checkpoints_baseline -> baseline)
        if 'checkpoints_' in checkpoint_path.parent.name:
            config_name = checkpoint_path.parent.name.replace('checkpoints_', '')
            output_dir = Path(f'evaluation_results_{config_name}')
        else:
            output_dir = Path('evaluation_results')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create image encoder
    image_encoder = CLIPImageEncoder(embedding_dim=512)
    image_encoder.load_state_dict(checkpoint['model_state_dict'])
    image_encoder.to(device)
    image_encoder.eval()
    
    # Load CLIP model (full model for get_text_features)
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
    text_encoder = clip_model.text_model  # For evaluation
    text_encoder.to(device)
    text_encoder.eval()
    clip_model.to(device)
    clip_model.eval()
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = COCODataset(
        images_dir=Path(args.data_root) / 'images' / 'val2014',
        cache_file=Path(args.cache_dir) / 'val_embeddings.pt',
        metadata_file=Path(args.cache_dir) / 'val_metadata.pt'
    )
    
    # Use subset if specified
    if args.subset_size:
        from torch.utils.data import Subset
        indices = random.sample(range(len(val_dataset)), min(args.subset_size, len(val_dataset)))
        val_dataset = Subset(val_dataset, indices)
        print(f"Using subset of {len(val_dataset)} samples")
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    metrics, similarity_matrix, img_emb, txt_emb = evaluate_model(
        image_encoder, text_encoder, val_loader, device, args.batch_size
    )
    
    # Print metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print("\nImage-to-Text Retrieval:")
    print(f"  Recall@1:  {metrics['recall@1_i2t']:.4f} ({metrics['recall@1_i2t']*100:.2f}%)")
    print(f"  Recall@5:  {metrics['recall@5_i2t']:.4f} ({metrics['recall@5_i2t']*100:.2f}%)")
    print(f"  Recall@10: {metrics['recall@10_i2t']:.4f} ({metrics['recall@10_i2t']*100:.2f}%)")
    print(f"  Mean Rank: {metrics['mean_rank_i2t']:.2f}")
    
    print("\nText-to-Image Retrieval:")
    print(f"  Recall@1:  {metrics['recall@1_t2i']:.4f} ({metrics['recall@1_t2i']*100:.2f}%)")
    print(f"  Recall@5:  {metrics['recall@5_t2i']:.4f} ({metrics['recall@5_t2i']*100:.2f}%)")
    print(f"  Recall@10: {metrics['recall@10_t2i']:.4f} ({metrics['recall@10_t2i']*100:.2f}%)")
    print(f"  Mean Rank: {metrics['mean_rank_t2i']:.2f}")
    print("="*70)
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_dir / 'metrics.json'}")
    
    # Visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Text-to-Image retrieval examples
    text_queries = ['sport', 'animal', 'person', 'food', 'vehicle']
    for query in text_queries:
        try:
            visualize_text_to_image_retrieval(
                query, image_encoder, clip_model, processor,
                val_dataset, device, top_k=5,
                save_path=output_dir / f'text_to_image_{query}.png'
            )
        except Exception as e:
            print(f"Error visualizing '{query}': {e}")
    
    # Image classification examples
    class_labels = ['a person', 'an animal', 'a landscape', 'food', 'a vehicle', 'sports equipment']
    
    # Get some random images
    sample_indices = random.sample(range(len(val_dataset)), min(5, len(val_dataset)))
    for idx in sample_indices:
        # Handle both regular dataset and Subset wrapper
        try:
            if hasattr(val_dataset, 'indices'):  # It's a Subset
                actual_idx = val_dataset.indices[idx]
                img_path = Path(args.data_root) / 'images' / 'val2014' / val_dataset.dataset.filenames[actual_idx]
            else:  # Regular dataset
                img_path = Path(args.data_root) / 'images' / 'val2014' / val_dataset.filenames[idx]
        except (AttributeError, IndexError):
            # Fallback: try to get filename directly
            try:
                img_path = Path(args.data_root) / 'images' / 'val2014' / val_dataset.filenames[idx]
            except:
                print(f"Warning: Could not get image path for index {idx}, skipping...")
                continue
        
        try:
            visualize_image_classification(
                img_path, image_encoder, clip_model, processor,
                class_labels, device,
                save_path=output_dir / f'classification_{idx}.png'
            )
        except Exception as e:
            print(f"Error classifying image {idx}: {e}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    # Import transforms
    from torchvision import transforms
    main()

