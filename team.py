from typing import List

from players import Player, PLAYERS


class Team:
    def __init__(self, p0: Player, p1: Player):
        self.players = [p0, p1]

    def __str__(self):
        return str(self.players[0]) + str(self.players[1])

    def has_player(self, p: Player) -> bool:
        """

        :param p:
        :return:
        """
        return p in self.players

    def get_players(self) -> List[Player]:
        """

        :return:
        """
        return self.players

    def get_teammate(self, p: Player) -> Player:
        """

        :param p:
        :return:
        """
        assert (p in self.players)
        return self.players[0] if p == self.players[1] else self.players[1]


TEAMS = [Team(PLAYERS[0], PLAYERS[2]),
         Team(PLAYERS[1], PLAYERS[3])]
