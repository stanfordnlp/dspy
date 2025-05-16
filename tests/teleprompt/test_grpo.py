from dspy.teleprompt.grpo import GRPO


def test_grpo_dataset_shuffler():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=3,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(4):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 3
        assert set(trainset_instances[-1]) == set(dataset)


def test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=2,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(15):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 2

    from collections import Counter

    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)

    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10


def test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset():
    dataset = [1, 2, 3]
    grpo = GRPO(
        num_dspy_examples_per_grpo_step=5,
        exclude_demos=True,
    )

    trainset_instances = []
    for i in range(6):
        trainset_instances.append(grpo.select_training_sample_and_update_shuffled_trainset(dataset, i))
        assert len(trainset_instances[-1]) == 5

    from collections import Counter

    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)

    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10


if __name__ == "__main__":
    test_grpo_dataset_shuffler()
    test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset()
    test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset()
    print("All tests passed!")
