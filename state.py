from copy import copy
from typing import Dict, List

from cards import Card
from players import PLAYERS_CYCLE, Player, Team
from trick import Trick


class State:
    """ Current state of the game of Bridge."""

    def __init__(self, trick: Trick,
                 teams: List[Team],
                 players: List[Player],
                 prev_tricks: List[Trick],
                 score: Dict[Team, int],
                 curr_player=None) -> None:
        self.trick = trick
        self.teams = teams
        self.players = players
        self.prev_tricks = prev_tricks
        self.score = score  # tricks won by teams
        self.curr_player = curr_player
        self.players_pos = {player.position: player for player in self.players}

        self.already_played = set()

    def get_successor(self, action: Card):
        """

        :param action: Card to play
        :returns State: Resulting state of game if playing `action`
        """
        assert (action in self.get_legal_actions())

        teams = [copy(self.teams[i]) for i in range(len(self.teams))]
        score = {teams[i]: self.score[team] for i, team in
                 enumerate(self.teams)}
        players = teams[0].get_players() + teams[1].get_players()
        trick = self.trick.create_from_other_players(players)
        curr_player = [p for p in players if p == self.curr_player][0]

        successor = State(trick, teams, players, self.prev_tricks, score,
                          curr_player)
        successor.apply_action(action)
        return successor

    def apply_action(self, card: Card, is_real_game:bool = False) -> None:
        """

        :param card: Action to apply on current state
        :param is_real_game: TODO [oriyan] Maryna, what is this?
        """
        assert (len(self.trick) < len(self.players_pos))
        assert card not in self.already_played

        prev_num_cards = len(self.curr_player.hand.cards)
        self.curr_player.play_card(card)
        curr_num_cards = len(self.curr_player.hand.cards)
        assert prev_num_cards != curr_num_cards

        self.trick.add_card(self.curr_player, card)
        self.already_played.add(card)
        assert self.already_played.isdisjoint(self.curr_player.hand.cards)

        if len(self.trick) == len(self.players_pos):  # last card played - open new trick
            if is_real_game:
                self.prev_tricks.append(copy(self.trick))
            winner_position = self.trick.get_winner()
            self.curr_player = self.players_pos[winner_position]
            i = 0 if self.teams[0].has_player(self.curr_player) else 1
            self.score[self.teams[i]] += 1
            self.trick = Trick({})
        else:
            assert self.curr_player in self.players_pos.values()
            assert self.curr_player in self.players
            # print(f"Mapping of position->next player: {repr(self.players_pos)}")
            self.curr_player = self.players_pos[PLAYERS_CYCLE[self.curr_player.position]]
            assert self.curr_player in self.players_pos.values()
            assert self.curr_player in self.players

    def get_legal_actions(self) -> List[Card]:
        legal_actions = self.curr_player.get_legal_actions(self.trick, self.already_played)
        assert self.already_played.isdisjoint(legal_actions)
        return legal_actions

    def get_score(self, curr_team_indicator) -> int:
        """ Returns score of team

        :param curr_team_indicator: [oriyan] is this for determining if player is max/min player? clarification needed
        :returns: current score of team
        """
        # assume there are 2 teams
        i, j = (0, 1) if self.teams[0].has_player(self.curr_player) else (1, 0)
        curr_team, other_team = self.teams[i], self.teams[j]
        if curr_team_indicator:
            return self.score[curr_team]
        return self.score[other_team]

    def __copy__(self):
        trick = copy(self.trick)
        prev_tricks = [copy(trick) for trick in self.prev_tricks]
        teams = [copy(team) for team in self.teams]
        score = {teams[0]: self.score[self.teams[0]],
                 teams[1]: self.score[self.teams[1]]}
        players = [copy(player) for player in self.players]
        curr_player_pos = self.curr_player.position
        state = State(trick, teams, players, prev_tricks, score, None)
        state.curr_player = state.players_pos[curr_player_pos]
        played = set(self.already_played)
        state.already_played = played
        return state

    @property
    def is_game_over(self) -> bool:
        for player in self.players:
            if len(player.hand.cards) != 0:
                return False
        return True
