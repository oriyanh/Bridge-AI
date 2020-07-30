from game import *
from typing import TextIO


PLAYERS_DICT = {'N': 0, 'E': 1, 'S': 2, 'W': 3}


class DataGame:
    def __init__(self, hands, first_player, tricks):
        self.hands = hands
        self.first_player = first_player
        self.tricks = tricks


class TrickValidation(Trick):
    def __init__(self):
        super().__init__()
        self.first_player = None

    def add_first_player(self, player: Player):
        self.first_player = player


def parse_file(file_name):
    games = []
    with open(file_name) as f:
        for line in f:
            if line[1:6] == "Deal ":
                hands = parse_hands(line[7:-3])
            elif line[1:5] == "Play":
                first_player, tricks = parse_game_steps(line, f)
                games.append(DataGame(hands, first_player, tricks))
    return games


def players_iterator(player_str: str):
    """
    Returns iterator of all players, starting with given player
    :param player_str: one letter from ['N', 'E', 'S', 'W']
    """
    cycle_players: Iterator[Player] = cycle(PLAYERS)
    curr_player = next(cycle_players)
    while curr_player.p.name != player_str:
        curr_player = next(cycle_players)
    return cycle_players


def parse_hands(line: str) -> List[Hand]:
    player_str, all_hands_str = line.split(':')
    cycle_players = players_iterator(player_str)

    hands = []
    hands_str = all_hands_str.split(' ')
    for hand in hands_str:
        cards_str = hand.split('.')
        cards = []
        for i, suit in enumerate(cards_str):
            for face in suit:
                cards.append(Card(face=face, suit=SUITS[i]))
        hands.append(Hand(next(cycle_players), cards))

    return hands


def parse_game_steps(player_line: str, f: TextIO):
    first_player = PLAYERS[PLAYERS_DICT[player_line[7]]]
    tricks = []

    cycle_players = players_iterator('W')
    trick_line = f.readline()
    while trick_line[0] != '*':
        curr_trick = TrickValidation()
        cards = trick_line.split(' ')
        if '-' in cards:  # last trick in game, some players give up
            break
        for c in cards:
            curr_trick.add_card(next(cycle_players), Card(face=c[1], suit=c[0]))
            curr_trick.add_first_player(first_player)
        first_player = curr_trick.get_winner()
        tricks.append(curr_trick)
        trick_line = f.readline()

    return first_player, tricks


def test_parse_file():
    # file = "bridge_data_1/Aulpll23.pbn"
    file = "bridge_data_1/Dkofro20.pbn"
    games = parse_file(file)
    return games


if __name__ == '__main__':
    test_parse_file()
