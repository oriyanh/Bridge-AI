from itertools import cycle
from numpy import argmax
from numpy.random import shuffle
import os
from typing import List, Iterator

from agents import *
from card import Card, FACES, SUITS
from hand import Hand
from players import PLAYERS
from team import TEAMS
from trick import Trick


class Game:
    def __init__(self,
                 agent: Agent,
                 other_agent: Agent,
                 games_counter: List[int],
                 verbose_mode: bool = True):

        self.agent = agent
        self.other_agent = other_agent
        self.games_counter = games_counter
        self.verbose_mode = verbose_mode

        self.tricks_counter: List[int] = [0, 0, ]  # [Team 0, Team 1]
        self.last_trick_winner: Player = choice(PLAYERS)
        self.cycle_players: Iterator[Player] = cycle(PLAYERS)
        self.winning_team: int = -1
        self.curr_trick = Trick()
        self.hands: Dict[Player, Hand] = self._deal_4_hands()

    def __str__(self):

        ret = ""

        ret += f"Match score: " \
               f"{TEAMS[0]}:{self.games_counter[0]:02} - " \
               f"{TEAMS[1]}:{self.games_counter[1]:02}\n"

        ret += f"Game score: " \
               f"{TEAMS[0]}:{self.tricks_counter[0]:02} - " \
               f"{TEAMS[1]}:{self.tricks_counter[1]:02}\n"

        ret += f"Current trick:  "
        for player, card in self.curr_trick.items():
            ret += f"{player}:{card}  "
        if len(self.curr_trick) == 4:
            ret += f",  {self.curr_trick.get_winner()} won trick."
        ret += f"\n"

        for player, hand in self.hands.items():
            ret += f"\n{player}\n{hand}"

        return ret

    @staticmethod
    def _deal_4_hands() -> Dict[Player, Hand]:
        """
        Generates a whole pack of cards, shuffles it, divides it into 4
        hands, and assigns them to the players.
        :return: A dict mapping a player to it's hand.
        """
        pack = []
        for f in FACES:
            for s in SUITS:
                pack.append(Card(f, s))
        shuffle(pack)

        hands = {}
        hand_len = len(pack) // len(PLAYERS)
        for i, player in enumerate(PLAYERS):
            hand_cards = pack[hand_len * i:hand_len * (i + 1)]
            hands[player] = Hand(player, hand_cards)

        return hands

    def play_hand(self, player: Player) -> None:
        """
        Called when its' the givens' player turn. The player will pick a
        card to play and it will be taken out of his hand a placed into the
        trick.
        :param player: The player who's turn it is.
        :return: None
        """
        if TEAMS[0].has_player(player):
            card = self.agent.get_action(
                player, self.hands, self.curr_trick)
        else:
            card = self.other_agent.get_action(
                player, self.hands, self.curr_trick)
        self.hands[player].cards.remove(card)
        self.curr_trick.add_card(player, card)

    def clear_trick(self) -> None:
        """
        Called once all 4 cards placed in trick. Decided winner, updates
        tricks winning counter, clears trick.
        :return: None
        """
        self.last_trick_winner = self.curr_trick.get_winner()

        if TEAMS[0].has_player(self.last_trick_winner):
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

            for i in range(len(PLAYERS)):  # Play all hands
                if i == 0:
                    curr_player = self.last_trick_winner
                else:
                    curr_player = next(self.cycle_players)
                self.play_hand(curr_player)

                if self.verbose_mode:
                    self.show()

            # Trick ended, calc result and reset trick.
            self.clear_trick()
            if self.verbose_mode:
                self.show()

        # Game ended, calc result.
        self.winning_team = int(argmax(self.tricks_counter))

    def show(self) -> None:
        """
        Update GUI
        :return: None
        """

        os.system('cls')
        print(self)
        input()
