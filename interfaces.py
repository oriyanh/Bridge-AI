from  abc import ABC, abstractmethod, abstractproperty
from typing import Dict


class IState(ABC):

    def __init__(self) -> None:
        self.trick = None
        self.players = None
        self.trump = None
        self.score = None

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