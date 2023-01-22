import tqdm
import pandas as pd

from IPython.display import display
from dsp.utils import EM

def evaluate(fn, dev, metric=EM):
    data = []

    for example in tqdm.tqdm(dev):
        question = example.question
        prediction = fn(question)

        d = dict(example)

        d['prediction'] = prediction
        d['correct'] = metric(prediction, example.answer)
        data.append(d)

    df = pd.DataFrame(data)

    print(f"Answered {df['correct'].sum()} / {len(dev)} correctly.")
    df['correct'] = df['correct'].apply(lambda x: '✔️' if x else '❌')

    pd.options.display.max_colwidth = None
    display(df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))
