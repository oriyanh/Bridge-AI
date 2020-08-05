from copy import copy
from players import PLAYERS_CYCLE
from trick import Trick


class State:
    def __init__(self, trick, teams, players, prev_tricks, score, curr_player=None) -> None:
        self.trick = trick
        self.teams = teams
        self.players = players
        self.prev_tricks = prev_tricks
        self.score = score  # tricks won by teams
        self.curr_player = curr_player

    def get_successors(self, action):
        assert(action in self.get_legal_actions())

        teams = [copy(self.teams[i]) for i in range(len(self.teams))]
        score = {teams[i]: self.score[team] for i, team in enumerate(self.teams)}
        players = teams[0].get_players() + teams[1].get_players()
        trick = self.trick.create_from_other_players(players)
        curr_player = [p for p in players if p == self.curr_player][0]

        successor = State(trick, teams, players, self.prev_tricks, score, curr_player)
        successor.apply_action(action)
        return successor

    def apply_action(self, card, is_real_game=False):
        players_pos = {player.position: player for player in self.players}
        assert (len(self.trick) < len(players_pos))
        self.curr_player.play_card(card)
        self.trick.add_card(self.curr_player, card)
        if len(self.trick) == len(players_pos):  # last card played - open new trick
            if is_real_game:
                self.prev_tricks.append(copy(self.trick))
            self.curr_player = self.trick.get_winner()
            i = 0 if self.teams[0].has_player(self.curr_player) else 1
            self.score[self.teams[i]] += 1
            self.trick = Trick({})
        else:
            self.curr_player = players_pos[PLAYERS_CYCLE[self.curr_player.position]]

    def get_legal_actions(self):
        return self.curr_player.get_legal_actions(self.trick)

    def get_score(self, curr_team_indicator):
        # assume there are 2 teams
        i, j = (0, 1) if self.teams[0].has_player(self.curr_player) else (1, 0)
        curr_team, other_team = self.teams[i], self.teams[j]
        if curr_team_indicator:
            return self.score[curr_team]
        return self.score[other_team]

    def is_end_game(self):
        if len(self.curr_player.hand) == 0:
            return True
        return False

