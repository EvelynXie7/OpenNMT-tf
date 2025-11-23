import os
import json
import pandas as pd

results = "results_0.jsonl"

df = pd.read_json(results, lines=True)

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
method_col = summary_df.pop('method')
summary_df.insert(0, 'method', method_col)
print(summary_df)
#
#final.to_csv("summary.csv", index=False)
