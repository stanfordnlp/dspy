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
            return self.trainset[:self.TRAIN_NUM]

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
        
    def get_evalset(self, EVAL_NUM=None):
        if EVAL_NUM:
            self.EVAL_NUM = EVAL_NUM
            return self.devset[:EVAL_NUM]
        else:
            return self.devset[:self.EVAL_NUM]

    def set_splits(self, TRAIN_NUM=None, DEV_NUM=None, EVAL_NUM=None):
        if TRAIN_NUM:
            self.TRAIN_NUM = TRAIN_NUM
        if DEV_NUM:
            self.DEV_NUM = DEV_NUM
        if EVAL_NUM:
            self.EVAL_NUM = EVAL_NUM