import os
import sys
import numpy as np
from copy import copy
from typing import List

from cards import Deck, TrumpType
from players import POSITIONS, Player, PositionEnum, TEAMS, Team
from state import State
from trick import Trick


class Game:

    def __init__(self,
                 agent,
                 other_agent,
                 games_counter: List[int],
                 tricks_counter: List[int],
                 verbose_mode: bool = True,
                 previous_tricks: List[Trick] = None,
                 curr_trick: Trick = None,
                 starting_pos: PositionEnum = None,
                 trump=None,
                 cards_in_hand=13):
        # todo(oriyan/maryna): think how to reproduce game from database -
        #  or randomly generate new game
        self.cards_in_hand=cards_in_hand
        self.agent = agent  # type: IAgent
        self.other_agent = other_agent  # type: IAgent
        self.games_counter = games_counter
        self.verbose_mode = verbose_mode
        if trump is None:
            trump = np.random.choice(TrumpType)
        else:
            trump = TrumpType.from_str(trump)
        self.trump = trump  # type: TrumpType
        self.deck = Deck(self.trump)
        hands = self.deck.deal(cards_in_hand=cards_in_hand)
        self.players = {pos: Player(pos, hand) for pos, hand in
                        zip(POSITIONS, hands)}
        self.teams = [Team(self.players[pos1], self.players[pos2]) for
                      pos1, pos2 in TEAMS]

        self.curr_trick = curr_trick
        self.previous_tricks = previous_tricks
        self.tricks_counter = tricks_counter
        self.winning_team: int = -1

        if starting_pos is None:
            starting_pos = np.random.choice(POSITIONS)
        self.curr_player = self.players[starting_pos]
        self._state = None

    def __str__(self):

        ret = ""

        ret += f"Match score: " \
               f"{self.teams[0]}:{self.games_counter[0]:02} - " \
               f"{self.teams[1]}:{self.games_counter[1]:02}\n"

        ret += f"Game score: " \
               f"{self.teams[0]}:{self.tricks_counter[0]:02} - " \
               f"{self.teams[1]}:{self.tricks_counter[1]:02}\n"
        ret += f"Trump Suite: {self.trump.value}\n"
        ret += f"Current trick:  "
        for player, card in self.curr_trick.items():
            ret += f"{player}:{card}  "
        if len(self.curr_trick) == 4:
            ret += f", {self.players[self.curr_trick.get_winner()]} won trick."
        ret += f"\n"

        for player in self.players.values():
            ret += f"\n{player}\n{player.hand}"

        return ret

    def run(self) -> bool:
        """
        Main game runner.
        :return: None
        """
        score = {self.teams[0]: 0, self.teams[1]: 0}
        initial_state = State(self.curr_trick, self.teams,
                              list(self.players.values()),
                              self.previous_tricks, score,
                              self.curr_player, trump=self.trump)
        self._state = initial_state
        self.previous_tricks = self._state.prev_tricks
        self.game_loop()
        return True

    def game_loop(self) -> None:
        while max(self.tricks_counter) < np.ceil(self.cards_in_hand / 2):  # Winner is determined.

            for i in range(len(POSITIONS)):  # Play all hands
                self.play_single_move()
                if self.verbose_mode:
                    self.show()
            if self.verbose_mode:
                self.show()

        # Game ended, calc result.
        self.winning_team = int(np.argmax(self.tricks_counter))

    def play_single_move(self) -> None:
        """
        Called when its' the givens' player turn. The player will pick a
        action to play and it will be taken out of his hand a placed into the
        trick.
        """
        if self.teams[0].has_player(self.curr_player):
            card = self.agent.get_action(self._state)
        else:
            card = self.other_agent.get_action(self._state)
        assert(card is not None)

        curr_trick = self._state.apply_action(card, True)
        self.curr_trick = curr_trick
        self.curr_player = self._state.curr_player  # Current player of state is trick winner
        self.tricks_counter = [self._state.score[self._state.teams[0]],
                               self._state.score[self._state.teams[1]]]

    def show(self) -> None:
        """
        Update GUI
        :return: None
        """

        os.system('clear' if 'linux' in sys.platform else 'cls')
        print(self)
        input()


class SimulatedGame(Game):
    """ Simulates a game with a non-empty state"""

    def __init__(self, agent, other_agent,
                 verbose_mode: bool = True, state: State = None, starting_action=None):
        """

        :param State state: Initial game state.
        :param Card starting_action: Initial play of current player.
            If None, chosen according to `agent`'s policy.
        """

        state_copy = copy(state)
        self.players = {player.position: player for player in state_copy.players}
        self.teams = state_copy.teams
        self.tricks_counter = [state_copy.score[state_copy.teams[0]],
                               state_copy.score[state_copy.teams[1]]]
        self.starting_action = starting_action
        self.first_play = True
        self.agent = agent  # type: IAgent
        self.other_agent = other_agent  # type: IAgent
        self.games_counter = [0, 0]
        self.verbose_mode = verbose_mode
        self.trump = state_copy.trump
        self.deck = Deck(self.trump)
        self.curr_trick = state_copy.trick
        self.previous_tricks = state_copy.prev_tricks
        self.winning_team: int = -1
        self.curr_player = state_copy.curr_player
        self._state = state_copy

    def play_single_move(self, get_card_only=False) -> None:
        if self.first_play and self.starting_action is not None:
            card = self.starting_action
            self.first_play = False
        elif self.teams[0].has_player(self.curr_player):
            card = self.agent.get_action(self._state)
        else:
            card = self.other_agent.get_action(self._state)

        if get_card_only:
            return card

        curr_trick = self._state.apply_action(card, True)
        self.curr_trick = curr_trick
        self.curr_player = self._state.curr_player  # Current player of state is trick winner
        self.tricks_counter = [self._state.score[self._state.teams[0]],
                               self._state.score[self._state.teams[1]]]

    def game_loop(self) -> None:
        if len(self.curr_trick.cards()) > 0:
            for card in self.curr_trick.cards():
                self._state.already_played.add(card)
        for _ in range(13 - len(self._state.prev_tricks)):
            for __ in range(4 - len(self.curr_trick.cards())):  # Play all hands
                self.play_single_move()
                if self.verbose_mode:
                    self.show()
            if max(self.tricks_counter) >= 7:  # Winner is determined.
                break
        self.winning_team = int(np.argmax(self.tricks_counter))

    def run(self) -> bool:
        self.game_loop()
        return True

