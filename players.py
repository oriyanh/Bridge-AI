from abc import ABC, abstractmethod
from enum import Enum

PlayersEnum = Enum("PlayersEnum", ['N', 'E', 'S', 'W'])


class Player:
    def __init__(self, p: PlayersEnum):
        self.p = p

    def __str__(self):
        return self.p.name

class PlayerNew:
    def __init__(self, p: PlayersEnum):
        self.p = p
        self.hand = None
        self.direction = None

    def play_card(self):
        pass
    def get_legal_actions(self, state):
        pass

    def __str__(self):
        return self.p.name


PLAYERS = [Player(PlayersEnum.N),
           Player(PlayersEnum.E),
           Player(PlayersEnum.S),
           Player(PlayersEnum.W)]
