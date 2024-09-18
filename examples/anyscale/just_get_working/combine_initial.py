import json
import os

def combine_initial_data():
    # get all files that start with "mini_checkpoint_trainset_data"
    files = [f for f in os.listdir() if f.startswith("mini_checkpoint_trainset_data")]

    combined_data = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            combined_data.extend(data)

    with open("combined_data.json", "w") as f:
        json.dump(combined_data, f)

    return combined_data

def deduplicate_data(data):

    data_pairs = [(data["prompt"], data["completion"]) for data in data]
    unique_data_pairs = list(set(data_pairs))
    return [{"prompt": pair[0], "completion": pair[1]} for pair in unique_data_pairs]

def read_combined_data():
    with open("combined_data.json", "r") as f:
        data = json.load(f)
        return data

if __name__ == "__main__":
    data = combine_initial_data()
    data = deduplicate_data(data)
    # data = read_combined_data()
    # data = [{"prompt": data["prompt"], "completion": data["completion"]} for data in data]
    train_data = data[:int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9):]
    with open("train_data.json", "w") as f:
        json.dump(train_data, f)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f)
    print(data[0])
    print(len(data))