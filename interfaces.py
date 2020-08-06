from abc import ABC, abstractmethod

from cards import Card
from state import State


class IState(ABC):

    def __init__(self) -> None:
        self.trick = None
        self.players = None
        self.trump = None
        self.score = None
        self.goal = None
        self.current_player = None

    def get_successors(self, legal_moves):
        pass

    def apply_action(self, action):
        pass

    def previous_tricks(self):
        pass

    def get_legal_actions(self):
        pass


class IAgent(ABC):
    @abstractmethod
    def __init__(self, target):
        self.target = target

    @abstractmethod
    def get_action(self, state: State) -> Card:
        """
        Pick a action to play based on the environment and a programmed
        strategy.
        :param state:
        :return: The action to play.
        """
        raise NotImplementedError
