import os
import json
import pandas as pd

root_dir = "/home/yuc3/OpenNMT-tf/evaluate/results_111725"

os.chdir(root_dir)

results = os.listdir(root_dir)

df = pd.concat([pd.read_json(result, lines=True) for result in results])

methods = df['method'].unique()

split_df = [df[df['method'] == method] for method in methods]

summary = []

for method_summary in split_df: 
    method = method_summary['method'].unique()
    results = method_summary['result'].reset_index().drop('index', axis=1)
    results = pd.concat([results.drop('result', axis = 1), results['result'].apply(pd.Series)], axis=1)
    n = len(results)
    average = results.sum() / n
    average_dict = json.loads(average.to_json())
    average_dict['method'] = method[0]
    summary.append(average_dict)
     
summary_df = pd.DataFrame(summary)
dropped_loss = summary_df.drop('loss', axis=1)
final = dropped_loss.drop('perplexity', axis=1)
method_col = final.pop('method')
final.insert(0, 'method', method_col)

final.to_csv("summary.csv", index=False)
