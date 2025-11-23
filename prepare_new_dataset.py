"""Download and prepare nmt-parallel-corpus (en-zh)."""

from datasets import load_dataset
from huggingface_hub import login
import os
import random
import argparse


def authenticate(token):
    """Authenticate with Hugging Face."""
    if token:
        login(token=token)
        return token
    
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token)
        return token
    
    return None


def save_split(pairs, prefix, output_dir):
    """Save pairs to EN and ZH files."""
    en_path = os.path.join(output_dir, f"{prefix}.en")
    zh_path = os.path.join(output_dir, f"{prefix}.zh")
    
    with open(en_path, 'w', encoding='utf-8') as f_en, \
         open(zh_path, 'w', encoding='utf-8') as f_zh:
        for en, zh in pairs:
            f_en.write(en.strip() + '\n')
            f_zh.write(zh.strip() + '\n')
    
    print(f"  ✓ {len(pairs):,} pairs → {prefix}.{{en,zh}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data/en-zh-new')
    parser.add_argument('--train_size', type=int, default=4900000)
    parser.add_argument('--valid_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    token = authenticate(args.token)
    
   
    dataset = load_dataset(
        "liboaccn/nmt-parallel-corpus",
        "en-zh",
        token=token,
        split='train'
   
    
    print(f"✓ Loaded {len(dataset):,} pairs")
    print(f"Columns: {dataset.column_names}")
    
    # Extract and shuffle
    total_needed = args.train_size + args.valid_size + args.test_size
    

    pairs = []
    skipped = 0
    
    for i, ex in enumerate(dataset):
        # Use correct column names: src_txt and tgt_txt
        en_text = ex['src_txt']
        zh_text = ex['tgt_txt']
        
        # Skip empty pairs
        if en_text and zh_text and len(en_text.strip()) > 0 and len(zh_text.strip()) > 0:
            pairs.append((en_text, zh_text))
        else:
            skipped += 1
        
        if len(pairs) >= total_needed:
            break
        
        if (i + 1) % 1000000 == 0:
            print(f"  Processed {i+1:,}, extracted {len(pairs):,} valid pairs (skipped {skipped:,})...")
    
    random.seed(args.seed)
    random.shuffle(pairs)

    
    # Split
    train_end = args.train_size
    valid_end = train_end + args.valid_size
    
    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]
    
    # Save
    
    
  
    save_split(train_pairs, 'train', args.output_dir)
    save_split(valid_pairs, 'valid', args.output_dir)
    save_split(test_pairs, 'test', args.output_dir)
    
    # Summary

    print(f"\nCreated in: {args.output_dir}/")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs):,} pairs")
    print(f"  Valid: {len(valid_pairs):,} pairs")
    print(f"  Test:  {len(test_pairs):,} pairs")
    
    return 0


if __name__ == "__main__":
    exit(main())