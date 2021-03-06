"""
run with the following arguments:
--agent1 <agent> --agent2 <agent> --num_games <int>

Where each agent encoding is of in one of the following forms:
Simple-<simple_agent_names>
AlphaBeta-<ab_evaluation_agent_names>-<depth>
MCTS-<'simple'/'stochastic'/'pure'>-<simple_agent_names>-<num_simulations>
Human

Optional arguments:
--name_games - defaults to 100
--verbose_mode - if 0, will only print end result of match with no user interaction. If 1, enter interactive mode.
--seed - if a number >= 0, set random seed of match for reproducibility. Else, use system default value.
"""

import os
import sys
import numpy as np
from argparse import ArgumentParser, ArgumentTypeError
from time import perf_counter
from tqdm import tqdm

from game import Game
from multi_agents import *
from trick import Trick


class Match:
    """ Represents a series of games of bridge, with same opponents."""

    def __init__(self,
                 agent: IAgent,
                 other_agent: IAgent,
                 num_games: int,
                 verbose_mode: bool = True,
                 cards_in_hand: int = 13):
        self.agent = agent
        self.other_agent = other_agent
        self.num_games = num_games
        self.verbose_mode = verbose_mode
        self.cards_in_hand = cards_in_hand

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
                                    self.games_counter, self.verbose_mode,
                                    cards_in_hand=self.cards_in_hand)
            curr_game.run()
            self.games_counter[curr_game.winning_team] += 1
        end_t = perf_counter()
        if self.verbose_mode:
            os.system('clear' if 'linux' in sys.platform else 'cls')
            print(self)
        print(self)
        print(f"Total time for match: {end_t - start_t} seconds; "
              f"Average {(end_t - start_t) / float(self.num_games)} "
              f"seconds per game")


def create_game(agent, other_agent, games_counter, verbose_mode,
                from_db=False, cards_in_hand=13):
    """ Returns Game object, either new random game or a game initialized from game DB"""
    if from_db:
        pass
    # todo(maryna): create single game from db. pay attention to players
    #  initialization + the iterator.
    trick_counter = [0, 0, ]  # [Team 0, Team 1]
    previous_tricks = []
    game = Game(agent, other_agent, games_counter, trick_counter, verbose_mode,
                previous_tricks, Trick({}), cards_in_hand=cards_in_hand)
    return game


def parse_args():
    """ Parses command line arguments. Returns namespace of arguments."""
    parser = ArgumentParser()
    parser.add_argument('--agent1', required=True)
    parser.add_argument('--agent2', required=True)
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--cards_in_hand', type=int, default=13)
    parser.add_argument('--verbose_mode', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)

    return parser.parse_args()


def str_to_agent(agent_str):
    agent_str = agent_str.split('-')
    if agent_str[0] == "Simple":  # Simple agent
        if agent_str[1] in simple_agent_names:
            return SimpleAgent(
                simple_func_names[simple_agent_names.index(
                    agent_str[1])])
        else:
            print("Bad arguments for Simple agent. Should be:\n"
                  "Simple-<agent name>")
            return -1

    elif agent_str[0] == "AlphaBeta":  # AlphaBeta agent
        if agent_str[1] in simple_agent_names:
            return AlphaBetaAgent(
                evaluation_function=ab_evaluation_func_names[
                    ab_evaluation_agent_names.index(agent_str[1])],
                depth=int(agent_str[2]))
        else:
            print("Bad arguments for AlphaBeta agent. Should be:\n"
                  "AlphaBeta-<ab_evaluation_agent_names>-<depth>")
            return -1

    elif agent_str[0] == "MCTS":  # MCTS agent
        if agent_str[1] == 'simple':
            return SimpleMCTSAgent(
                action_chooser_function=simple_func_names[
                    simple_agent_names.index(agent_str[2])],
                num_simulations=int(agent_str[3]))
        elif agent_str[1] == 'stochastic':
            return StochasticSimpleMCTSAgent(
                action_chooser_function=simple_func_names[
                    simple_agent_names.index(agent_str[2])],
                num_simulations=int(agent_str[3]))
        elif agent_str[1] == 'pure':
            return PureMCTSAgent(
                action_chooser_function=simple_func_names[
                    simple_agent_names.index(agent_str[2])],
                num_simulations=int(agent_str[3]))
        else:
            print("Bad arguments for MCTS agent. Should be:\n"
                  "MCTS-<'simple'/'stochastic'/'pure'>-"
                  "<simulated agent name>-<num_of_simulations>")
            return -1

    elif agent_str[0] == "Human":  # Human agent
        return HumanAgent()

    else:
        raise ArgumentTypeError()


def run_match():
    try:
        a0 = str_to_agent(args.agent1)
        a1 = str_to_agent(args.agent2)
    except ArgumentTypeError:
        print("ArgumentTypeError: Bad arguments usage", file=sys.stderr)
        print(f"USAGE: run with the following arguments - ", file=sys.stderr)
        print("   python3.7 match.py --agent1 <agent> --agent2 <agent> [--cards_in_hand <int> --num_games <int> --verbose_mode <int> --seed <int>]\n"
                "Where each agent encoding is of in one of the following forms:\n"
                "* Simple-<simple_agent_names>\n"
                "* AlphaBeta-<ab_evaluation_agent_names>-<depth>\n"
                "* MCTS-<'simple'/'stochastic'/'pure'>-<simple_agent_names>-<num_simulations>\n"
                "* Human", file=sys.stderr)
        exit(1)

    match = Match(agent=a0,
                  other_agent=a1,
                  num_games=args.num_games,
                  verbose_mode=bool(args.verbose_mode),
                  cards_in_hand=args.cards_in_hand)
    match.run()


if __name__ == '__main__':
    args = parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
    run_match()
    input("Press Enter button to exit")
