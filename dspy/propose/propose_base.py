from abc import ABC, abstractmethod


class Proposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def propose_instructions_for_program(self):
        pass

    def propose_instruction_for_predictor(self):
        pass
