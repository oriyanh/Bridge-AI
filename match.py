import os
import sys
from argparse import ArgumentParser
from numpy.random import seed
from time import perf_counter
from tqdm import tqdm

from game import Game
from multi_agents import *
from trick import Trick

NUM_GAMES = 3

seed(0)


class Match:
    """ Represents a series of games of bridge, with same opponents."""

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

        start_t = perf_counter()
        for _ in tqdm(range(self.num_games),
                      leave=False, disable=self.verbose_mode, file=sys.stdout):
            curr_game = create_game(self.agent, self.other_agent,
                                    self.games_counter, self.verbose_mode)
            curr_game.run()
            self.games_counter[curr_game.winning_team] += 1
        end_t = perf_counter()
        if self.verbose_mode:
            os.system('clear' if 'linux' in sys.platform else 'cls')
            print(self)
        print(self)
        print(f"Total time for match: {end_t - start_t} seconds; Average {(end_t-start_t)/float(self.num_games)} seconds per game")


def create_game(agent, other_agent, games_counter, verbose_mode,
                from_db=False):
    """ Returns Game object, either new random game or a game initialized from game DB"""
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
    """ Parses command line arguments. To be implemented."""
    parser = ArgumentParser()
    parser.add_argument('--agent1')
    parser.add_argument('--agent2')
    parser.add_argument('--rounds', default=100)
    args = parser.parse_args()
    return args


def run_match():
    match = Match(PureMCTSAgent('hard_greedy_action', num_simulations=100), SimpleAgent('soft_greedy_action'), NUM_GAMES, verbose_mode=False)
    match.run()


if __name__ == '__main__':
    run_match()
    input()
