from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_program(self):
        pass

    @abstractmethod
    def get_metric(self):
        pass

    def get_trainset(self, TRAIN_NUM=None):
        if TRAIN_NUM:
            self.TRAIN_NUM = TRAIN_NUM
            return self.trainset[:TRAIN_NUM]
        else:
            return self.trainset[: self.TRAIN_NUM]

    def get_devset(self, TRAIN_NUM=None, DEV_NUM=None):
        if TRAIN_NUM:
            self.TRAIN_NUM = TRAIN_NUM
        if DEV_NUM:
            self.DEV_NUM = DEV_NUM

        index = -1
        if DEV_NUM and TRAIN_NUM:
            index = max(len(self.trainset) - DEV_NUM, TRAIN_NUM)
        else:
            index = max(len(self.trainset) - self.DEV_NUM, self.TRAIN_NUM)
        return self.trainset[index:]

    def get_testset(self, TEST_NUM=None):
        if TEST_NUM:
            self.TEST_NUM = TEST_NUM
            return self.testset[:TEST_NUM]
        else:
            return self.testset[: self.TEST_NUM]

    def set_splits(self, TRAIN_NUM=None, DEV_NUM=None, TEST_NUM=None):
        if TRAIN_NUM:
            self.TRAIN_NUM = TRAIN_NUM
        if DEV_NUM:
            self.DEV_NUM = DEV_NUM
        if TEST_NUM:
            self.TEST_NUM = TEST_NUM

    def get_max_tokens(self):
        return 150
