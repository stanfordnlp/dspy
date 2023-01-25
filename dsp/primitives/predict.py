import dsp
from collections import Counter
from dsp.utils import zipstar, normalize_text


class Completions:
    def __init__(self, completions: list, template):
        self.data = completions
        self.template = template

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def unpack(self, key=None):
        if key:
            return [getattr(c, key) for c in self.data]

        return zipstar(self.data)

    def __getattr__(self, name):
        assert len(self.data) == 1

        completion = self.data[0]

        if name in completion.keys():
            return getattr(completion, name)
        
        if name.endswith('s') and name[:-1] in completion.keys():
            pass

        assert False, name



def generate(template, **kwargs):
    generator = dsp.settings.lm

    def do_generate(example, stage, max_depth=2):
        assert stage is not None

        # Look up the appropriate fields in each demonstration.
        example = example.demos_at(lambda d: d[stage])
        
        # Generate and extract the fields.
        prompt = template(example)
        completions = generator(prompt, **kwargs)
        completions = [template.extract(example, p) for p in completions]

        # Find the completions that are most complete.
        field_names = [field.input_variable for field in template.fields]

        last_field_idx = 0
        for field_idx, key in enumerate(field_names):
            completions_ = [c for c in completions if key in c.keys() and c[key] is not None]

            # Filter out completions that are missing fields that are present in at least one completion.
            if len(completions_):
                completions = completions_
                last_field_idx = field_idx+1
        
        # If none of the completions is completed (i.e., none has the final field set).
        if last_field_idx < len(field_names):
            # Pick the first completion that has gone farthest.
            completion = completions[0]
            completion[field_names[last_field_idx]] = ''

            # Recurse with greedy decoding and a shorter length.
            max_tokens = kwargs.get('max_tokens', dsp.settings.lm.kwargs['max_tokens'])
            max_tokens = min(max(75, max_tokens // 2), max_tokens)
            new_kwargs = {**kwargs, 'max_tokens': max_tokens, 'n': 1, 'temperature': 0.0}

            assert max_depth > 0
            return generate(template, **new_kwargs)(completion, stage=stage, max_depth=max_depth-1)

        completions = Completions(completions, template=template)
        example = example.copy(completions=completions)

        if len(completions) == 1:
            completion = completions[0]
            example[stage] = example.copy(**completion)

        return example, completions
    
    return do_generate


def generate_sc(example, prompt, normalize=True, extract=None, prediction_field=None, **kwargs):
    kwargs = {'temperature': 0.7, 'max_tokens': 150, 'n': 20, **kwargs}
    completions = dsp.settings.lm(prompt, **kwargs)
    completions = extract_final_answer(example, completions, extract=extract)
    return majority_vote_(completions, normalize=normalize, prediction_field=prediction_field)

def extract_final_answer(example, completions, extract=None):
    if extract:
        completions = [extract(example, p) for p in completions]
    else:
        completions = [p.strip().split('\n')[-1].split(':', 1)[-1].strip() for p in completions]

    dsp.settings.lm.history.append({**dsp.settings.lm.history[-1], 'completions': completions})

    return completions

def majority(completions, normalize=True, field=None):
    field = completions.template.fields[-1].output_variable if field is None else field

    return Completions(majority_vote_(completions, normalize=normalize, prediction_field=field),
                      template=completions.template)


def majority_vote_(completions, normalize: bool, prediction_field: str):
    if normalize:
        original_completions = completions
        completions_ = []
        for pred in completions:
            if prediction_field in pred:
                completions_.append(normalize_text(pred[prediction_field]))
            else:
                completions_.append('')
        completions = completions_
        normalized_to_original = {}

        for completion, normalized_completion in zip(original_completions, completions):
            if normalized_completion not in normalized_to_original:
                normalized_to_original[normalized_completion] = completion

    completions_ = [x for x in completions if x]
    # completions_ = [x for x in completions_ if len(x.split()) < 7]

    if completions_:
        completions = completions_

    topk = Counter(completions).most_common()
    pred, _ = topk[0]

    if normalize:
        pred = normalized_to_original[pred]

    dsp.settings.lm.history.append({**dsp.settings.lm.history[-1], 'topk': topk, 'completions': [pred]})

    return [pred]


