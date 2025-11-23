import evaluate
import tensorflow as tf
import torch

from multiprocessing import Process, Queue

import gc
import os
import json
from io import BytesIO
from contextlib import redirect_stdout
import random

os.chdir("/home/yuc3/OpenNMT-tf")
assert os.path.abspath(os.curdir) == "/home/yuc3/OpenNMT-tf"

# make sure that i'm using the correct opennmt python files
from opennmt import load_model, load_config, merge_config, Runner
from opennmt.utils.scorers import make_scorers
from opennmt.utils.decoding import NucleusSampler, DynamicBeamSearch

import numpy as np

def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

working_dir = "/home/yuc3/OpenNMT-tf"

# load model
model_yml = f"{working_dir}/config.enzh.v3.yml"
model_config = load_config([model_yml])
model_dir = f"{working_dir}/{model_config['model_dir']}"

# list of scoring methods to use to evaluage
evaluation_dir = "/home/yuc3/OpenNMT-tf/evaluate"
eval_yml = f"{evaluation_dir}/scorers.yml"

# reference file 
valid_zh_tok = f"{working_dir}/{model_config['data']['eval_labels_file']}"
#valid_zh_tok = f"{working_dir}/data/en-zh-v3/valid3.zh.tok"
references = read_lines(valid_zh_tok)

# feature file 
valid_en = f"{working_dir}/{model_config['data']['eval_features_file']}"
#valid_en = f"{working_dir}/data/en-zh-v3/valid3.en"

# where to store and read inference files from
inference_dir = f"{working_dir}/data/en-zh-v3/inferences"
os.makedirs(inference_dir, exist_ok=True)

# decoding strategies 
dec_ymls = [ 
    f"{evaluation_dir}/greedy.yml",     # this as base metric
    f"{evaluation_dir}/top_3.yml",
    f"{evaluation_dir}/top_5.yml",
    f"{evaluation_dir}/top_7.yml",
    f"{evaluation_dir}/top_p_02.yml",
    f"{evaluation_dir}/top_p_1.yml",
    f"{evaluation_dir}/top_p_50.yml",
    f"{evaluation_dir}/beam_3.yml",
    f"{evaluation_dir}/beam_5.yml",
    f"{evaluation_dir}/beam_7.yml",
    f"{evaluation_dir}/inc_beam_2.yml",
    f"{evaluation_dir}/inc_beam_4.yml",
    f"{evaluation_dir}/inc_beam_6.yml",
    f"{evaluation_dir}/dec_beam_2.yml",
    f"{evaluation_dir}/dec_beam_4.yml",
    f"{evaluation_dir}/dec_beam_6.yml"
]

random.shuffle(dec_ymls)

dec_configs = [
    (yml.split('/')[-1].split('.')[0], load_config([eval_yml, yml])) 
    for yml in dec_ymls
]

# In order to run different models on the same python script, I had to 
# encapsulate them in their own processes, so that GPU memory was released 
# so that the model intialization fails with OOM error
def tf_stuff(model_dir, model_config, config, feature_file, inference_file): 
    print("="*70)
    print("Running Model...")
    print("="*70)
    print(config)
    runner = Runner(load_model(model_dir), merge_config(model_config, config))
    runner.infer(valid_en, predictions_file=inference_file)
    print("Saved inferece at", inference_file)
    print("="*70)
    print("Saved inferece at", inference_file)
    print("="*70)

def torch_stuff(inferences, references, q):
    print("="*70)
    print("Running BERT score...")
    print("="*70)
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=inferences, references=references, lang="zh")
    q.put(results)

num_runs = 15
for i in range(0, num_runs): 
    with open(f"{evaluation_dir}/results_final.jsonl", "a") as file: 
        for decoding_method, config in dec_configs: 
            summary = {"method" : decoding_method}
            summary["result"] = {}

            # infer using the decoding_method
            inference_file = f"{inference_dir}/infer_{i}.out.{decoding_method}" 
            process_model = Process(target=tf_stuff, args=(model_dir, model_config, config, valid_en, inference_file,))
            process_model.start()
            process_model.join()
            inferences = read_lines(inference_file)
            print(decoding_method, inferences[0])

            print("="*70)
            print("EVALUATION RESULTS")
            print("="*70)

            # score the inferences  
            opennmt_metrics = ["bleu", "chrf"]
            scorers = make_scorers(opennmt_metrics)
            for scorer in scorers:
                score = scorer(valid_zh_tok, inference_file)
                if isinstance(score, dict):
                    for name, value in score.items():
                        summary["result"][name] = float(value)
                        print(f"{name}: {value:.4f}")
                else:
                    summary["result"][scorer.name] = float(score)
                    print(f"{scorer.name}: {score:.4f}")

            q = Queue()
            process_bert = Process(target=torch_stuff, args=(inferences, references, q))
            process_bert.start()
            results = q.get()
            process_bert.join()

            summary["result"]["bertscore-precision"] = np.mean(results['precision']).item()
            summary["result"]["bertscore-recall"] = np.mean(results['recall']).item()
            summary["result"]["bertscore-f1"] = np.mean(results['f1']).item()
            print(f"bertscore-precision: {np.mean(results['precision']):.4f}")
            print(f"bertscore-recall: {np.mean(results['recall']):.4f}")
            print(f"bertscore-f1: {np.mean(results['f1']):.4f}")

            summary_str = json.dumps(summary)
            file.write(f"{summary_str}\n")
            file.flush()
            print(summary_str)
