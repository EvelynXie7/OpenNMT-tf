""" BERT_score.py
    Adapt BERT Scorer from Huggingface library"""
import argparse
from opennmt.utils.scorers import make_scorers
import evaluate
import numpy as np

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--hypothesis', required=True)
    parser.add_argument('--metrics', nargs='+', default=['bleu', 'bertscore'],
                       help='Metrics to compute')
    parser.add_argument('--lang', default='en', help='Language for BERTScore')
    args = parser.parse_args()
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    
    opennmt_metrics = [m for m in args.metrics if m != 'bertscore']
    if opennmt_metrics:
        scorers = make_scorers(opennmt_metrics)
        for scorer in scorers:
            score = scorer(args.reference, args.hypothesis)
            if isinstance(score, dict):
                for name, value in score.items():
                    print(f"{name}: {value:.4f}")
            else:
                print(f"{scorer.name}: {score:.4f}")
    
 
    if 'bertscore' in args.metrics:
        print("\nComputing BERTScore...")
        bertscore = evaluate.load("bertscore")
        refs = read_lines(args.reference)
        hyps = read_lines(args.hypothesis)
        
        results = bertscore.compute(
            predictions=hyps,
            references=refs,
            lang=args.lang
        )
        
        print(f"bertscore-precision: {np.mean(results['precision']):.4f}")
        print(f"bertscore-recall: {np.mean(results['recall']):.4f}")
        print(f"bertscore-f1: {np.mean(results['f1']):.4f}")

if __name__ == "__main__":
    main()
