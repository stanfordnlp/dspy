# import os
# import random
# import subprocess
# import time

# import tqdm
# import ujson
# from datasets.fingerprint import Hasher

# import dsp

# if os.environ.get('DSP_NOTEBOOK_CACHEDIR'):
#     training_data_directory = os.path.join(os.environ.get('DSP_NOTEBOOK_CACHEDIR'), 'compiler')
# else:
#     training_data_directory = 'cache/compiler'


# compilations_assumed_to_exist={'ft-zvEdzQVQ5xwlxvNPrxl6kpnw': 'ada:ft-stanfordpraglab-2023-02-09-19-50-49'}


# def openai_check_finetune(jobname):
#     if dsp.settings.force_reuse_cached_compilation and jobname in compilations_assumed_to_exist:
#         return compilations_assumed_to_exist[jobname]

#     command = f"""openai api fine_tunes.get -i {jobname}"""
#     print(command)

#     result = subprocess.run(command.split(), stdout=subprocess.PIPE, check=False)
#     output = result.stdout.decode("utf-8").strip()

#     try:
#         output = ujson.loads(output)
#         if output['status'] == 'succeeded':
#             return output['fine_tuned_model']

#         if output['status'] in ['pending', 'running']:
#             print(f'Compiling, run ```openai api fine_tunes.follow -i {jobname}``` for details...')
#             time.sleep(60)
#             return openai_check_finetune(jobname)
#     except:
#         pass

#     return False


# def convert_to_training_point2(y, inputs, outputs, template):
#     assert len(inputs) + len(outputs) == len(template.fields)

#     y_ = dsp.Example(**{f: y[f] for f in inputs}, demos=[])
#     prompt = template(y_, show_guidelines=False)

#     completion = y[outputs[0]]
#     output_fields = template.fields[len(inputs):]

#     for field in output_fields[1:]:
#         completion += f"\n\n{field.name} " + y[field.output_variable]
    
#     completion = " " + completion + " </s>"
#     return {'prompt': prompt, 'completion': completion}


# def simulate(program, input_examples):
#     training_data = []

#     for input_example in tqdm.tqdm(input_examples):
#         prediction = program(input_example)

#         if prediction is not None:
#             # assert len(prediction.compiling_stages) == 2, "TMP"
#             for stage in prediction.compiling_stages:
#                 name, template, inputs, outputs = stage['name'], stage['template'], stage['inputs'], stage['outputs']
#                 training_data.append(convert_to_training_point2(prediction.get(name), inputs, outputs, template))
    
#     r = random.Random(0)
#     r.shuffle(training_data)

#     return training_data


# def openai_finetune_(name, target):
#     training_data_path = name_to_path(name)

#     # Launch the fine-tune on the path
#     command = f"""openai api fine_tunes.create -t {training_data_path} -m {target} --n_epochs 4 --learning_rate_multiplier 0.05 --no_check_if_files_exist"""
#     print(command)

#     # command = """python script.py"""
#     process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     while line := process.stdout.readline().decode().strip():
#         if 'created fine-tune:' in line.lower():
#             jobname = line.split()[-1]
#             break
        
#     #     if 'costs $' in line.lower():
#     #         cost = line.split()[-1]
#     #         break

#     # assert cost[0] == '$'
    
#     # if float(cost[1:]) > 300:
#     #     print(f'Got cost {cost} -- you may wanna cancel the job: openai api fine_tunes.cancel -i {jobname}')

#     # print(cost)

#     print(jobname)

#     # Block until it's done
#     ft = openai_check_finetune(jobname)
#     assert ft, ft

#     # Return its name
#     return (jobname, ft)


# def openai_finetune(name, target):
#     print(name)
#     training_data_path = name_to_path(name)
#     training_data_path += '.model'

#     # if path + stuff exists, load the tuple from it
#     try:
#         with open(training_data_path) as f:
#             jobname, ft = ujson.loads(f.readline())

#         if openai_check_finetune(jobname):
#             return jobname, ft
#     except:
#         pass
    
#     jobname, ft = openai_finetune_(name, target)

#     with open(training_data_path, 'w') as f:
#         f.write(ujson.dumps((jobname, ft)) + '\n')
    
#     return jobname, ft


# def name_to_path(name):
#     if not os.path.exists(training_data_directory):
#         os.makedirs(training_data_directory)

#     training_data_path = os.path.join(training_data_directory, f'{name}.jsonl')
#     return training_data_path


# # 3. Check that the output file name has status "success" (not deleted or non-existent). Otherwise, re-call with n = n+1.
# def finetune(training_data, target):
#     name = Hasher.hash(training_data)
#     training_data_path = name_to_path(name)

#     with open(training_data_path, 'w') as f:
#         for line in training_data:
#             f.write(ujson.dumps(line) + '\n')

#     jobname, ft = openai_finetune(name, target)
#     print(ft)

#     ft = dsp.GPT3(model=ft, stop=" </s>")
#     return ft

# # 4. Return updated program.
# def compile(program, examples, target='ada'):
#     training_data = simulate(program, examples)
#     compiled_lm = finetune(training_data, target=target)

#     def compiled_program(*args, **kwargs):
#         with dsp.settings.context(compiled_lm=compiled_lm, compiling=False):
#             return program(*args, **kwargs)

#     compiled_program.lm = compiled_lm
#     return compiled_program

