import os
from typing import List

from cards import Deck
from multi_agents import *
from players import POSITIONS, Player, PositionEnum, TEAMS, Team
from state import State
from trick import Trick


class Game:

    def __init__(self,
                 agent: IAgent,
                 other_agent: IAgent,
                 games_counter: List[int],
                 tricks_counter: List[int],
                 verbose_mode: bool = True,
                 previous_tricks: List[Trick] = None,
                 curr_trick: Trick = None,
                 starting_pos: PositionEnum = None):
        # todo(oriyan/maryna): think how to reproduce game from database -
        #  or randomly generate new game
        self.agent = agent
        self.other_agent = other_agent
        self.games_counter = games_counter
        self.verbose_mode = verbose_mode

        self.deck = Deck()
        hands = self.deck.deal()
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

        ret += f"Current trick:  "
        for player, card in self.curr_trick.items():
            ret += f"{player}:{card}  "
        if len(self.curr_trick) == 4:
            ret += f", {self.players[self.curr_trick.get_winner()]} won trick."
        ret += f"\n"

        for player in self.players.values():
            ret += f"\n{player}\n{player.hand}"

        return ret

    def run(self, initial_state=None) -> None:
        """
        Main game runner.
        :return: None
        """
        if initial_state is None:
            score = {self.teams[0]: 0, self.teams[1]: 0}
            initial_state = State(self.curr_trick, self.teams,
                                  list(self.players.values()),
                                  self.previous_tricks, score,
                                  self.curr_player)
        self._state = initial_state
        self.game_loop()

    def game_loop(self) -> None:
        while max(self.tricks_counter) < 13 // 2 + 1:  # Winner is determined.

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

        self._state.apply_action(card, True)
        self.curr_trick = self._state.trick
        self.curr_player = self._state.curr_player  # Current player of state is trick winner
        self.tricks_counter = [self._state.score[self._state.teams[0]],
                               self._state.score[self._state.teams[1]]]

    def show(self) -> None:
        """
        Update GUI
        :return: None
        """

        os.system('cls')
        print(self)
        input()
