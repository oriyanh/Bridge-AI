from itertools import cycle
import numpy as np
import os
from typing import List, Iterator

from agents import *
from cards import Deck
from players import POSITIONS, Player, PositionEnum, TEAMS, Team
from trick import Trick


class Game:

    def __init__(self,
                 agent: Agent,
                 other_agent: Agent,
                 games_counter: List[int],
                 verbose_mode: bool = True,
                 starting_pos=None):
        # todo [oriyan\mar] think of life, its meaning, and everything
        # todo also, think how to reproduce game from database - or randomly generate new game
        self.agent = agent
        self.other_agent = other_agent
        self.games_counter = games_counter
        self.verbose_mode = verbose_mode
        self.deck = Deck()

        self.tricks_counter: List[int] = [0, 0, ]  # [Team 0, Team 1]

        self.winning_team: int = -1
        self.curr_trick = Trick()
        self.previous_tricks = []  # type: List[Trick]
        hands = self.deck.deal()
        self.players = {position: Player(position, hand) for position, hand in zip(POSITIONS, hands)}
        self.teams = [Team(self.players[pos1], self.players[pos2]) for pos1, pos2 in TEAMS]
        self.last_trick_winner: PositionEnum = np.random.choice(POSITIONS) if not starting_pos else starting_pos
        self.cycle_players: Iterator[PositionEnum] = cycle(POSITIONS)

    def __str__(self):

        ret = ""

        ret += f"Match score: " \
               f"{self.teams[0]}:{self.games_counter[0]:02} - " \
               f"{self.teams[1]}:{self.games_counter[1]:02}\n"

        ret += f"Game score: " \
               f"{self.teams[0]}:{self.tricks_counter[0]:02} - " \
               f"{self.teams[1]}:{self.tricks_counter[1]:02}\n"

        ret += f"Current trick:  "
        for player, card in self.curr_trick.items():
            ret += f"{player}:{card}  "
        if len(self.curr_trick) == 4:
            ret += f",  {self.curr_trick.get_winner()} won trick."
        ret += f"\n"

        for player in self.players.values():
            ret += f"\n{player}\n{player.hand}"

        return ret

    def play(self, player: Player) -> None:
        """
        Called when its' the givens' player turn. The player will pick a
        card to play and it will be taken out of his hand a placed into the
        trick.
        :param player: The player who's turn it is.
        :return: None
        """
        if self.teams[0].has_player(player):
            # TODO [oriyan/mar] Instead of passing hands and trick to get_action, pass State object
            card = self.agent.get_action(player, self.players, self.curr_trick)

        else:
            card = self.other_agent.get_action(
                player, self.players, self.curr_trick)
        player.play_card(card)  # TODO [oriyan/mar]
        self.curr_trick.add_card(player, card)

    def clear_trick(self) -> None:
        """
        Called once all 4 cards placed in trick. Decided winner, updates
        tricks winning counter, clears trick.
        :return: None
        """
        self.last_trick_winner = self.curr_trick.get_winner()

        if self.teams[0].has_player(self.players[self.last_trick_winner]):
            self.tricks_counter[0] += 1  # Team 0 won trick
        else:
            self.tricks_counter[1] += 1  # Team 1 won trick
        self.curr_trick.reset()

    def run(self) -> None:
        """
        Main game runner.
        :return: None
        """

        while max(self.tricks_counter) < 13 // 2 + 1:  # Winner is determined.

            # Set next opening player according to last trick results.
            while next(self.cycle_players) != self.last_trick_winner:
                pass

            for i in range(len(POSITIONS)):  # Play all hands
                if i == 0:
                    curr_player = self.last_trick_winner
                else:
                    curr_player = next(self.cycle_players)
                player = self.players[curr_player]
                self.play(player)

                if self.verbose_mode:
                    self.show()

            # Trick ended, calc result and reset trick.
            self.clear_trick()
            if self.verbose_mode:
                self.show()

        # Game ended, calc result.
        self.winning_team = int(np.argmax(self.tricks_counter))

    def show(self) -> None:
        """
        Update GUI
        :return: None
        """

        os.system('cls')
        print(self)
        input()
