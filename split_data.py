import random

def split_data(data, train_size=0.8, test_size=0.2):
    """The data is divided into training set and test set.
    Args:
        data: A list containing all data.
        train_size: Training set to validation set ratio.  (The validation set is separated from this training data during model training)
        test_size: Test set proportion.
    """
    if sum([train_size, test_size]) != 1.0:
        raise ValueError("The sum of the proportions of the training set and the test set must be 1.0")

    data = data.copy()
    random.shuffle(data)  # randomize data

    train_idx = int(len(data) * train_size)

    train_data = data[:train_idx]
    test_data = data[train_idx:]

    return train_data, test_data

with open("data/data1.txt", "r", encoding="utf-8") as f:
    all_data = f.read().splitlines()

train_data, test_data = split_data(all_data, train_size=0.8, test_size=0.2)

with open("data/data1_train.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(train_data))

with open("data/data1_test.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(test_data))

print(f"finish！")
print(f"1: {len(train_data)}")
print(f"test: {len(test_data)}")