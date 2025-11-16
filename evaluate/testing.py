from opennmt import load_model, load_config, merge_config, Runner
import os
import json

working_dir = "/home/xiee/NLP/OpenNMT-tf"

os.chdir(working_dir)

# load model
model_yml = f"{working_dir}/config.enzh.v3.yml"
model_config = load_config([model_yml])
model_dir = f"{working_dir}/{model_config['model_dir']}"
model = load_model(model_dir)

# list of scoring methods to use to evaluage
evaluation_dir = "/home/yuc3/OpenNMT-tf/evaluate"
eval_yml = f"{evaluation_dir}/scorers.yml"

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

dec_configs = [
    (yml.split('/')[-1].split('.')[0], load_config([eval_yml, yml])) 
    for yml in dec_ymls
]
print(dec_configs)
runners = {
    name: Runner(model, merge_config(model_config, config)) 
    for name, config in dec_configs
}
config_1 = load_config([eval_yml, dec_ymls[-1]])
config_2 = load_config([eval_yml, dec_ymls[8]])
print(Runner(model, merge_config(model_config, config_1)).evaluate())
print(Runner(model, merge_config(model_config, config_2)).evaluate())

results = {}

for decoding_method in runners: 
    results[name] = runners[decoding_method].evaluate()


#results["greedy"] = runner_greedy.evaluate()
#results["top_5"] = runner_top_5.evaluate()
#results["beam"] = runner_beam.evaluate()
#results["inc_beam_2"] = runner_inc_beam_2.evaluate()
#results["inc_beam_4"] = runner_inc_beam_4.evaluate()
#results["inc_beam_6"] = runner_inc_beam_6.evaluate()
#
#print(results)

with open("results.json", "w") as json_file: 
    json.dump(results, json_file, indent=4)
