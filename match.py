from numpy.random import seed
from os import system
from sys import stdout
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser

from agents import *
from game import Game

NUM_GAMES = 3

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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('agent1')
    parser.add_argument('agent2')
    parser.add_argument('--rounds', default=100)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    game = Match(RandomAgent(), SoftGreedyAgent(), NUM_GAMES)
    game.run()
    input()
