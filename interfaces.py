from  abc import ABC, abstractmethod, abstractproperty
from typing import Dict


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
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, player: IPlayer, state: IState) -> Card:
        """
        Pick a card to play based on the environment and a programmed strategy.
        :param player: Current player
        :param hands: Current hands in play
        :param trick: Current trick in play
        :return: The card to play.
        """
        raise NotImplementedError