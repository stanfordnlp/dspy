---
sidebar_position: 3
---

# Creating a Custom Dataset

We saw until now how you can play around with `Example` objects and how you can use `HotPotQA` class to load HotPotQA dataset from HuggingFace to a list of `Example` object. But in production, rarely you'll see yourself working on such dataset. Rather you'll find yourself working on a custom dataset and you might question, how do I create my own dataset or what format should my dataset be in?

In DSPy, you need your dataset to be a list of `Examples`. So if your dataset has 1000 datapoints you'll need to transform that in a list of 1000 `Example` objects. Based on this fact we can do it in two ways:

* **The Pythonic Way:** Using native python utility and logic.
* **The DSPythonic Way:** Using DSPy's `Dataset` class.

## The Pythonic Way

We essentially want to create a list of `Example` objects, so all we need to do is load the data from the data source like Comma-Seperated Values(CSV), HuggingFace, etc. and formulate it into a list of `Example` via a basic loop or list comprehension. Let's take an example CSV `sample.tsv` that contains 3 fields: **context**, **question** and **summary**. Let's start by loading it via Pandas:

```python
import pandas as pd

df = pd.read_csv("sample.csv")
print(df.shape)
```
**Output:**
```text
(1000, 3)
```

Now all we need to do is iterate over each row in `df` and append it to a list after adding data into an `Example` object.

```python
dataset = []

for context, question, answer in df.values:
    dataset.append(dspy.Example(context=context, question=question, answer=answer).with_inputs("context", "question"))

print(dataset[:3])
```
**Output:**
```python
[Example({'context': nan, 'question': 'Which is a species of fish? Tope or Rope', 'answer': 'Tope'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': 'Why can camels survive for long without water?', 'answer': 'Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?", 'answer': 'The name of the third daughter is Alice'}) (input_keys={'question', 'context'})]
```

And there we go we have a dataset ready to be used, pretty simple right? Pretty minimalistic too, what if we do even less work? That's what the DSPythonic way will help us with!!

## The DSPythonic Way

In the Pythonic Way, we loaded the dataset and appended it in a list after converting it into a `Example` object. But there is an even easier way to do this by using `Dataset` class, using which we'll just need to load the dataset and rest of the processing will be taken care by class methods as we saw in the previous article. So to summarize what we'll be doing is:

* Load data from CSV to a dataframe.
* Split the data to train, dev and test splits.
* Populate `_train`, `_dev` and `_test` class attributes. Note that these attributes should be a list of dictionary, or an iterator over mapping like HuggingFace Dataset, to make it work.

All the above stuff can be done in `__init__` method, meaning in the most minimalistic form `__init__` is the only method we need to implement.

```python
import pandas as pd
from dspy.datasets.dataset import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        df = pd.read_csv(file_path)
        self._train = df.iloc[0:700].to_dict(orient='records')

        self._dev = df.iloc[700:].to_dict(orient='records')

dataset = CSVDataset("sample.csv")
print(dataset.train[:3])
```
**Output:**
```text
[Example({'context': nan, 'question': 'Which is a species of fish? Tope or Rope', 'answer': 'Tope'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': 'Why can camels survive for long without water?', 'answer': 'Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time.'}) (input_keys={'question', 'context'}),
 Example({'context': nan, 'question': "Alice's parents have three daughters: Amy, Jessy, and what’s the name of the third daughter?", 'answer': 'The name of the third daughter is Alice'}) (input_keys={'question', 'context'})]
```

Let's understand the code step by step:

* It subclasses the base `Dataset` class from DSPy. This inherits all the useful data loading/processing functionality.
* We load the data in CSV into a DataFrame.
* We get the **train** split i.e first 700 rows in the DataFrame and convert it to lists of dicts using `to_dict(orient='records')` method and is then assigned to `self._train`.
* We get the **dev** split i.e first 300 rows in the DataFrame and convert it to lists of dicts using `to_dict(orient='records')` method and is then assigned to `self._dev`.

Using the Dataset base class makes loading custom datasets incredibly easy. Load the data and populate the `self._train`, `self._dev`, `self._test`. The base `Dataset` class will handle converting the data into Example objects and shuffling/sampling the data. This avoids having to write all that boilerplate code ourselves for every new dataset.

:::caution

We did not populate `_test` attribute in the above code, which is fine and won't cause any unneccesary error as such. However it'll give you an error if you try to access the test split.

```python
dataset.test[:5]
```
****
```text
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-59-5202f6de3c7b> in <cell line: 1>()
----> 1 dataset.test[:5]

/usr/local/lib/python3.10/dist-packages/dspy/datasets/dataset.py in test(self)
     51     def test(self):
     52         if not hasattr(self, '_test_'):
---> 53             self._test_ = self._shuffle_and_sample('test', self._test, self.test_size, self.test_seed)
     54 
     55         return self._test_

AttributeError: 'CSVDataset' object has no attribute '_test'
```

To prevent that you'll just need to make sure `_test` is not `None` and populated with the approproate data.

:::

You can overide the methods in `Dataset` class to customize you class even more. So in summary, the Dataset base class provides a clean way to load and preprocess custom datasets in a consistent manner with minimal code!
