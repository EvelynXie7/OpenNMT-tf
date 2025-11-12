"""Download and prepare nmt-parallel-corpus (en-zh).

Creates three splits:
- train: for training (goes in config)
- valid: for validation during training (goes in config as eval_*)
- test: for final evaluation after training (NOT in config, used separately)
"""

from datasets import load_dataset
import os
import random
import argparse


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
    parser.add_argument('--train_size', type=int, default=2000000)
    parser.add_argument('--valid_size', type=int, default=200000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--token', type=str, default=None)
    args = parser.parse_args()
    
    token = args.token or os.environ.get('HF_TOKEN')
    os.makedirs(args.output_dir, exist_ok=True)
    

    if not token:
        print("⚠️  Gated dataset - authenticate first:")
        print("   huggingface-cli login\n")
    
    try:
        dataset = load_dataset(
            "liboaccn/nmt-parallel-corpus",
            "en-zh",
            token=token,
            split='train'
        )
    except Exception as e:
  
        print("\nRun: huggingface-cli login")
        return 1
    
    print(f"✓ Loaded {len(dataset):,} pairs")
    
    # Extract and shuffle
    total_needed = args.train_size + args.valid_size + args.test_size  
    
    print(f"\nExtracting {total_needed:,} pairs...")
    pairs = []
    for i, ex in enumerate(dataset):
        pairs.append((ex['en'], ex['zh']))
        if len(pairs) >= total_needed:
            break
        if (i + 1) % 500000 == 0:
            print(f"  {i+1:,}...")
    
    print(f"✓ Extracted {len(pairs):,} pairs")
    
    random.seed(args.seed)
    random.shuffle(pairs)
    print(f"✓ Shuffled (seed={args.seed})")
    
    # Split
    train_end = args.train_size
    valid_end = train_end + args.valid_size
    
    train_pairs = pairs[:train_end]
    valid_pairs = pairs[train_end:valid_end]
    test_pairs = pairs[valid_end:]
    
    # Save
   
    
    print("\n✏️  TRAIN (used in config: train_features_file/train_labels_file)")
    save_split(train_pairs, 'train', args.output_dir)
    
    print("\n✏️  VALID (used in config: eval_features_file/eval_labels_file)")
    save_split(valid_pairs, 'valid', args.output_dir)
    
    print("\n✏️  TEST (NOT in config - used after training for final eval)")
    save_split(test_pairs, 'test', args.output_dir)
    
    # Summary
   
    print(f"\nCreated in: {args.output_dir}/")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_pairs):,} pairs")
    print(f"  Valid: {len(valid_pairs):,} pairs")
    print(f"  Test:  {len(test_pairs):,} pairs")
    print(f"\n📝 Config should reference:")
    print(f"   train_features_file: {args.output_dir}/train.en")
    print(f"   train_labels_file:   {args.output_dir}/train.zh")
    print(f"   eval_features_file:  {args.output_dir}/valid.en")
    print(f"   eval_labels_file:    {args.output_dir}/valid.zh")
    print(f"\n📊 After training, evaluate on test:")
    print(f"   onmt-main --config config.yml infer \\")
    print(f"     --features_file {args.output_dir}/test.en \\")
    print(f"     --predictions_file outputs/test_pred.zh")
    
    return 0


if __name__ == "__main__":
    exit(main())