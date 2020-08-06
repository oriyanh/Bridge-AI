from argparse import ArgumentParser
from os import system
from sys import stdout
from typing import List
from argparse import ArgumentParser

from numpy.random import seed
from tqdm import tqdm

from game import Game
from multi_agents import *
from trick import Trick

NUM_GAMES = 3

seed(0)


class Match:
    def __init__(self,
                 agent: IAgent,
                 other_agent: IAgent,
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
            curr_game = create_game(self.agent, self.other_agent,
                                    self.games_counter, self.verbose_mode)
            curr_game.run()
            self.games_counter[curr_game.winning_team] += 1

        if self.verbose_mode:
            system('cls')
            print(self)


def create_game(agent, other_agent, games_counter, verbose_mode,
                from_db=False):
    if from_db:
        pass
    # todo(maryna): create single game from db. pay attention to players
    #  initialization + the iterator.
    trick_counter = [0, 0, ]  # [Team 0, Team 1]
    previous_tricks = []
    game = Game(agent, other_agent, games_counter, trick_counter, verbose_mode,
                previous_tricks, Trick({}))
    return game


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--agent1')
    parser.add_argument('--agent2')
    parser.add_argument('--rounds', default=100)
    args = parser.parse_args()
    return args


def run_match():
    match = Match(SimpleAgent(), SimpleAgent('soft_greedy_action'), NUM_GAMES)
    match.run()


if __name__ == '__main__':
    run_match()
    input()
