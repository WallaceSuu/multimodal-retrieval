"""
Interactive Text-to-Image Search

This script allows users to input text queries and retrieve matching images
from the validation set using the trained CLIP model.
"""

import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_clip import CLIPImageEncoder, COCODataset


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create image encoder
    image_encoder = CLIPImageEncoder(embedding_dim=512)
    image_encoder.load_state_dict(checkpoint['model_state_dict'])
    image_encoder.to(device)
    image_encoder.eval()
    
    # Load CLIP model for text encoding
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
    clip_model.to(device)
    clip_model.eval()
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    print("Model loaded successfully!")
    return image_encoder, clip_model, processor


def search_images(text_query, image_encoder, clip_model, processor, dataset, device, top_k=5):
    """
    Search for images matching a text query.
    
    Args:
        text_query: Text string to search for
        image_encoder: Trained image encoder
        clip_model: CLIP model for text encoding
        processor: CLIP processor
        dataset: Dataset to search in
        device: Device to run on
        top_k: Number of top images to retrieve
        
    Returns:
        List of (image_path, similarity_score) tuples
    """
    image_encoder.eval()
    clip_model.eval()
    
    # Encode text query
    with torch.no_grad():
        inputs = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True).to(device)
        query_embedding = clip_model.get_text_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    
    # Compute similarities with all images
    print(f"Searching for: '{text_query}'...")
    similarities = []
    image_paths = []
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for images, _, filenames in tqdm(data_loader, desc="Computing similarities"):
            images = images.to(device)
            image_embeddings = image_encoder(images)
            
            # Compute similarity
            sim = torch.matmul(query_embedding, image_embeddings.t())
            similarities.append(sim.cpu())
            
            # Store image paths
            image_paths.extend([dataset.images_dir / f for f in filenames])
    
    similarities = torch.cat(similarities, dim=1).squeeze(0)
    
    # Get top-K indices
    top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    # Return results
    results = []
    for idx in top_k_indices:
        img_idx = idx.item()
        img_path = image_paths[img_idx]
        similarity = similarities[img_idx].item()
        results.append((img_path, similarity))
    
    return results


def visualize_results(text_query, results, save_path=None):
    """
    Visualize top-K retrieved images.
    
    Args:
        text_query: The search query
        results: List of (image_path, similarity_score) tuples
        save_path: Optional path to save the visualization
    """
    top_k = len(results)
    fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
    if top_k == 1:
        axes = [axes]
    
    for idx, (img_path, similarity) in enumerate(results):
        ax = axes[idx]
        
        # Load and display image
        try:
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            ax.set_title(f'Rank {idx+1}\nSimilarity: {similarity:.3f}', fontsize=10)
        except Exception as e:
            ax.text(0.5, 0.5, f'Image not found\n{img_path.name}', 
                   ha='center', va='center', fontsize=8)
        ax.axis('off')
    
    plt.suptitle(f'Top-{top_k} Image Retrievals for: "{text_query}"', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    plt.close()


def main():
    """Main interactive function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Text-to-Image Search')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_baseline/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='coco2014',
                       help='COCO data root')
    parser.add_argument('--cache_dir', type=str, default='cache',
                       help='Cache directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top images to retrieve')
    parser.add_argument('--save_dir', type=str, default='search_results',
                       help='Directory to save search results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    image_encoder, clip_model, processor = load_model(args.checkpoint, device)
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = COCODataset(
        images_dir=Path(args.data_root) / 'images' / 'val2014',
        cache_file=Path(args.cache_dir) / 'val_embeddings.pt',
        metadata_file=Path(args.cache_dir) / 'val_metadata.pt'
    )
    print(f"Loaded {len(val_dataset)} validation images\n")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Interactive loop
    print("="*70)
    print("TEXT-TO-IMAGE SEARCH")
    print("="*70)
    print("Enter text queries to search for matching images.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user input
        text_query = input("Enter search query: ").strip()
        
        if text_query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if not text_query:
            print("Please enter a valid query.\n")
            continue
        
        try:
            # Search for images
            results = search_images(
                text_query, image_encoder, clip_model, processor,
                val_dataset, device, top_k=args.top_k
            )
            
            # Display results
            print(f"\nTop {args.top_k} results for '{text_query}':")
            for rank, (img_path, similarity) in enumerate(results, 1):
                print(f"  {rank}. {img_path.name} (similarity: {similarity:.4f})")
            
            # Visualize
            save_path = save_dir / f'search_{text_query.replace(" ", "_")}.png'
            visualize_results(text_query, results, save_path=save_path)
            
            print()
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == '__main__':
    main()

