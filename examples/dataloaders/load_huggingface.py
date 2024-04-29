import datasets

from dspy.datasets.dataloader import DataLoader

if __name__ == "__main__":
    dl = DataLoader()

    total = 0
    correct = 0
    n = 100

    for dataset_name in datasets.list_datasets():
        total += 1
        try:
            info = datasets.get_dataset_infos(dataset_name)
            config = list(info.keys())[0]
            input_keys = tuple(info[config].features.keys())
            splits = [f"{split}[:{n}]" for split in list(info[config].splits.keys())]
            dataset = dl.from_huggingface(dataset_name, input_keys=input_keys, name=config, split=splits)
            correct += 1
            del dataset
        except Exception as e:
            print(f"{dataset_name}: {e}")

        print(f"Correct/Total: {correct/total}, {correct}/{total}")
