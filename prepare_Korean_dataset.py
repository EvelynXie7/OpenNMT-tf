from datasets import load_dataset
import multiprocess
from random import sample, shuffle 
from functools import partial 

num_cores = 25
num_samples = 25000

def chunks(ds, num_batches): 
    num_rows = ds.num_rows
    batch_size = num_rows // num_batches

    for i in range(num_batches):
        ds_chunk = ds[i * batch_size: (i + 1) * batch_size]
        yield list(zip(ds_chunk['tgt_txt'], ds_chunk['src_txt'])) 

def shuffle_sample(k, pairs): 
    shuffle(pairs)
    return sample(pairs, k)

ds = load_dataset("liboaccn/nmt-parallel-corpus", "en-ko", cache_dir="~/en_ko", split='train')

chunks = list(chunks(ds, num_cores))

with multiprocess.Pool(processes=num_cores) as pool: 
    k = num_samples // num_cores 
    shuffle_sample_k = partial(shuffle_sample, k)
    chunks_sampled = pool.map(shuffle_sample_k, chunks)

flattened = []
for chunk in chunks_sampled:
    flattened.extend(chunk)

with open('train.ko', 'w') as train_ko, open('train.en', 'w') as train_en: 
    for pair in flattened: 
        train_ko.write(pair[0])
        train_ko.write("\n")
        train_en.write(pair[1])
        train_en.write("\n")
