import json


def combine_initial_data():
    files = [
        "mini_checkpoint_trainset_data_1500_1500_1150_0_checkpoint_2.json",
        "mini_checkpoint_trainset_data_1500_1500_1310_0_checkpoint_1.json",
        "mini_checkpoint_trainset_data_1500_1500_1500_0_checkpoint_0.json",
    ]

    combined_data = []
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            combined_data.extend(data)

    with open("combined_data.json", "w") as f:
        json.dump(combined_data, f)

def read_combined_data():
    with open("combined_data.json", "r") as f:
        data = json.load(f)
        return data

if __name__ == "__main__":
    # combine_initial_data()
    data = read_combined_data()
    data = [{"prompt": data["prompt"], "completion": data["completion"]} for data in data]
    train_data = data[:int(len(data) * 0.9)]
    val_data = data[int(len(data) * 0.9):]
    with open("train_data.json", "w") as f:
        json.dump(train_data, f)
    with open("val_data.json", "w") as f:
        json.dump(val_data, f)
    # print(data[0])
    # print(len(data))