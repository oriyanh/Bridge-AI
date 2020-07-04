from numpy.random import seed
from os import system
from sys import stdout
from tqdm import tqdm
from typing import List

from agents import *
from game import Game

seed(0)


class Match:
    def __init__(self,
                 agent: Agent,
                 other_agent: Agent,
                 num_games: int,
                 verbose_mode: bool = True):
        self.agent = agent
        self.other_agent = other_agent
        self.num_games = num_games
        self.verbose_mode = verbose_mode

        self.games_counter: List[int] = [0, 0, ]  # [Team 0, Team 1]

    def __str__(self):
        ret = ""

        ret += f"Total score: " \
               f"{self.games_counter[0]:02} - {self.games_counter[1]:02}\n"

        return ret

    def run(self) -> None:
        """
        Main match runner.
        :return: None
        """
        for _ in tqdm(range(self.num_games),
                      leave=False, disable=self.verbose_mode, file=stdout):
            curr_game = Game(self.agent, self.other_agent,
                             self.games_counter, self.verbose_mode)
            curr_game.run()
            self.games_counter[curr_game.winning_team] += 1

        if self.verbose_mode:
            system('cls')
            print(self)


if __name__ == '__main__':
    game = Match(RandomAgent(), SoftGreedyAgent(), 10)
    game.run()
    input()
