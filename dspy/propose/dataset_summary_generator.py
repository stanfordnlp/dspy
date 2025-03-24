import re

import dspy
from dspy.propose.utils import strip_prefix

class ObservationSummarizer(dspy.Signature):
    ("""Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.""")
    observations = dspy.InputField(desc="Observations I have made about my dataset")
    summary = dspy.OutputField(desc="Two to Three sentence summary of only the most significant highlights of my observations")

class DatasetDescriptor(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")
    
    examples = dspy.InputField(desc="Sample data points from the dataset")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed")

class DatasetDescriptorWithPriorObservations(dspy.Signature):
    ("""Given several examples from a dataset please write observations about trends that hold for most or all of the samples. """
    """I will also provide you with a few observations I have already made.  Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE' """
    """Some areas you may consider in your observations: topics, content, syntax, conciceness, etc. """
    """It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative""")
    
    examples = dspy.InputField(desc="Sample data points from the dataset")
    prior_observations = dspy.InputField(desc="Some prior observations I made about the data")
    observations = dspy.OutputField(desc="Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add")

def order_input_keys_in_string(unordered_repr):
    # Regex pattern to match the input keys structure
    pattern = r"input_keys=\{([^\}]+)\}"

    # Function to reorder keys
    def reorder_keys(match):
        # Extracting the keys from the match
        keys_str = match.group(1)
        # Splitting the keys, stripping extra spaces, and sorting them
        keys = sorted(key.strip() for key in keys_str.split(','))
        # Formatting the sorted keys back into the expected structure
        return f"input_keys={{{', '.join(keys)}}}"

    # Using re.sub to find all matches of the pattern and replace them using the reorder_keys function
    ordered_repr = re.sub(pattern, reorder_keys, unordered_repr)

    return ordered_repr

def create_dataset_summary(trainset, view_data_batch_size, prompt_model, log_file=None, verbose=False):
    if verbose:
        print("\nBootstrapping dataset summary (this will be used to generate instructions)...")
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = prompt_model if prompt_model else dspy.settings.lm
    with dspy.settings.context(lm=prompt_model):
        observation = dspy.Predict(DatasetDescriptor, n=1, temperature=1.0)(examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__()))
    observations = observation["observations"]

    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")

    skips = 0
    try:
        max_calls = 10
        calls = 0
        for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
            calls+=1
            if calls >= max_calls:
                break
            if verbose:
                print(f"b: {b}")
            upper_lim = min(len(trainset), b+view_data_batch_size)
            with dspy.settings.context(lm=prompt_model):
                output = dspy.Predict(DatasetDescriptorWithPriorObservations, n=1, temperature=1.0)(prior_observations=observations, examples=order_input_keys_in_string(trainset[b:upper_lim].__repr__()))
            if len(output["observations"]) >= 8 and output["observations"][:8].upper() == "COMPLETE":
                skips += 1
                if skips >= 5:
                    break
                continue
            observations += output["observations"]
            
            if log_file: 
                log_file.write(f"observations {observations}\n")
    except Exception as e:
        if verbose:
            print(f"e {e}. using observations from past round for a summary.")

    if prompt_model:
        with dspy.settings.context(lm=prompt_model):
            summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(observations=observations)
    else:
        summary = dspy.Predict(ObservationSummarizer, n=1, temperature=1.0)(observations=observations)
    if verbose:
        print(f"summary: {summary}")
    if log_file:
        log_file.write(f"summary: {summary}\n")
    
    if verbose:
        print(f"\nGenerated summary: {strip_prefix(summary.summary)}\n")

    return strip_prefix(summary.summary)
