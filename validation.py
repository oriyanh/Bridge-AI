from cards import Card, SUITS, Hand, Deck, SUITS_ALT
from players import Player, POSITIONS, PLAYERS_CYCLE, TEAMS
from trick import Trick
from typing import Tuple, List, TextIO
from enum import Enum
from copy import copy


TEAMS_CYCLE = {TEAMS[0]: TEAMS[1], TEAMS[1]: TEAMS[0]}
PLAYERS_DICT = {'N': 0, 'E': 1, 'S': 2, 'W': 3}


class TrickValidation(Trick):
    """
    Class of Trick, with added characteristic of order of cards given by
    remembering the starting player.
    """
    def __init__(self):
        super().__init__({})
        self.first_player = None

    def add_first_player(self, player: Enum):
        self.first_player = player


class DataGame:
    """
    Class which stores a recorded game, with option to take a snapshot from any
    point in it.
    """
    def __init__(self, players: List[Player], tricks: List[TrickValidation],
                 winner: Tuple[Enum], trump: str):  # todo: trump to Suit?
        self.players = players
        self.tricks = tricks
        self.winner = winner
        self.trump = trump

    def snapshot(self, trick_idx: int, player: Enum):
        """
        Image of one moment in game, in trick_idx trick, when player should play
        :param trick_idx: first trick is 0
        :param player: the player to commit its turn now
        :return: current hands situation and trick on desk
        """
        if trick_idx >= len(self.tricks):
            raise IndexError(f"trick_idx argument has to be smaller then "
                             f"{len(self.tricks)}")

        # Load initial hands situation
        curr_hands = {}
        for p in self.players:
            curr_hands[p.position] = copy(p.hand)

        # Remove cards from all last tricks from hands
        for i in range(trick_idx):
            for p in POSITIONS:  # todo: change Tricks to get Enum, not Player
                curr_hands[p].play_card(self.tricks[i].trick[p])

        # Remove cards of current trick from hands of all players which play
        # before given player. In addition, store these cards
        curr_player = self.tricks[trick_idx].first_player  # todo: first player Enum
        curr_trick = Trick({})
        while curr_player != player:
            curr_hands[curr_player].play_card(
                self.tricks[trick_idx].trick[curr_player])
            curr_trick.add_card(
                curr_player, self.tricks[trick_idx].trick[curr_player])
            curr_player = PLAYERS_CYCLE[curr_player]

        return [curr_hands[p] for p in POSITIONS], curr_trick


def parse_file(file_name: str) -> List[DataGame]:
    """
    Get a path of PBN file, and parse it to list of DataGame objects
    """
    games = []
    with open(file_name) as f:
        for line in f:
            if line[1:6] == "Deal ":
                players = parse_players(line[7:-3])
            elif line[1:9] == "Declarer":
                win_team, trump = parse_winners_and_trump(line, f)
            elif line[1:5] == "Play":
                tricks = parse_tricks(line, f)
                games.append(DataGame(players, tricks, win_team, trump))
    return games


def parse_players(line: str) -> List[Player]:
    """
    Helper for parse_file
    :param line: line from PBN file, which starts with "[Deal "
    :return: list of 4 Hand objects of a game
    """
    player_str, all_hands_str = line.split(':')
    next_position = POSITIONS[PLAYERS_DICT[player_str]]

    players = []
    players_str = all_hands_str.split(' ')
    for p in players_str:
        curr_position = next_position
        cards_str = p.split('.')
        cards = []
        for i, suit in enumerate(cards_str):
            for face in suit:
                cards.append(Card(face=face, suit=SUITS[i]))
                next_position = PLAYERS_CYCLE[curr_position]
        players.append(Player(curr_position, Hand(cards)))

    return players


def parse_winners_and_trump(line: str, f: TextIO) -> Tuple[Tuple[Enum], str]:
    declarer = POSITIONS[PLAYERS_DICT[line[11]]]
    declare_team = TEAMS[0] if declarer in TEAMS[0] else TEAMS[1]

    bid_line = f.readline()
    obligation, trump = int(bid_line[11]) + 6, SUITS_ALT[bid_line[12]]

    res_line = f.readline()
    result = int(res_line[8:-2].split('\"')[1])
    win_team = declare_team if result >= obligation else TEAMS_CYCLE[declare_team]
    return win_team, trump


def parse_tricks(player_line: str, f: TextIO) -> List[TrickValidation]:
    """
    Helper for parse_file
    :param player_line: line from PBN file, which starts with "["Play"
    :param f: TextIO object, of the PBN file that read
    :return: list of all tricks of a game
    """
    first_position = POSITIONS[PLAYERS_DICT[player_line[7]]]
    curr_player = POSITIONS[0]
    tricks = []

    trick_line = f.readline()
    while trick_line[0] != '*':
        curr_trick = TrickValidation()
        cards = trick_line.split(' ')
        if '-' in cards:  # last trick in game, some players give up
            break
        for c in cards:  # todo: change add_card() to get Enum, not player
            curr_trick.add_card(curr_player, Card(face=c[1], suit=c[0]))
            curr_player = PLAYERS_CYCLE[curr_player]
            curr_trick.add_first_player(first_position)
        tricks.append(curr_trick)
        first_position = curr_trick.get_winner()
        trick_line = f.readline()

    return tricks


def test_snapshot():
    file = "bridge_data_1/Aulpll23.pbn"
    # file = "bridge_data_1/Dkofro20.pbn"
    games = parse_file(file)
    # out = games[0].snapshot(0, PLAYERS[0])
    # out = games[0].snapshot(0, PLAYERS[2])
    # out = games[0].snapshot(1, PLAYERS[2])
    # out = games[0].snapshot(1, PLAYERS[0])
    # out = games[0].snapshot(6, PLAYERS[0])
    # return out


if __name__ == '__main__':
    test_snapshot()
