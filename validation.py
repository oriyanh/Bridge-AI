from game import *
from typing import TextIO
from itertools import islice


PLAYERS_DICT = {'N': 0, 'E': 1, 'S': 2, 'W': 3}


class TrickValidation(Trick):
    def __init__(self):
        super().__init__()
        self.first_player = None

    def add_first_player(self, player: Player):
        self.first_player = player


class DataGame:
    def __init__(self, hands: List[Hand], tricks: List[TrickValidation]):
        self.hands = hands
        self.tricks = tricks

    def snapshot(self, trick_idx: int, player: Player):
        """
        Image of one moment in game, in trick_idx trick, when player should play
        :param trick_idx: first trick is 0
        :param player: the player to commit its turn now
        :return: current hands situation and trick on desk
        """
        if trick_idx >= len(self.tricks):
            raise IndexError(f"trick_idx has to be smaller then {len(self.tricks)}")

        # Load initial hands situation
        curr_hands = {}
        for h in self.hands:
            curr_hands[h.player] = h.cards

        # Remove cards from all last tricks from hands
        for i in range(trick_idx):
            for p in PLAYERS:
                curr_hands[p].remove(self.tricks[i].trick[p])

        # Remove cards of current trick from hands of all players which play
        # before given player. In addition, store these cards
        player_idx = PLAYERS_DICT[self.tricks[trick_idx].first_player.p.name]
        cycle_players = islice(cycle(PLAYERS), player_idx, None)
        curr_player = next(cycle_players)
        curr_trick = Trick()
        while curr_player != player:
            curr_hands[curr_player].remove(
                self.tricks[trick_idx].trick[curr_player])
            curr_trick.add_card(
                curr_player, self.tricks[trick_idx].trick[curr_player])
            curr_player = next(cycle_players)

        return [curr_hands[p] for p in PLAYERS], curr_trick


def parse_file(file_name) -> List[DataGame]:
    games = []
    with open(file_name) as f:
        for line in f:
            if line[1:6] == "Deal ":
                hands = parse_hands(line[7:-3])
            elif line[1:5] == "Play":
                tricks = parse_tricks(line, f)
                games.append(DataGame(hands, tricks))
    return games


def parse_hands(line: str) -> List[Hand]:
    player_str, all_hands_str = line.split(':')
    # Iterator for all players, starts in player_str
    cycle_players = islice(cycle(PLAYERS), PLAYERS_DICT[player_str], None)

    hands = []
    hands_str = all_hands_str.split(' ')
    for hand in hands_str:
        cards_str = hand.split('.')
        cards = []
        player = next(cycle_players)
        for i, suit in enumerate(cards_str):
            for face in suit:
                cards.append(Card(face=face, suit=SUITS[i]))
        hands.append(Hand(player, cards))

    return hands


def parse_tricks(player_line: str, f: TextIO) -> List[TrickValidation]:
    first_player = PLAYERS[PLAYERS_DICT[player_line[7]]]
    tricks = []

    # Iterator for all players, starts in 'N'
    cycle_players = islice(cycle(PLAYERS), 0, None)
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

    return tricks


def test_snapshot():
    file = "bridge_data_1/Aulpll23.pbn"
    # file = "bridge_data_1/Dkofro20.pbn"
    games = parse_file(file)
    # out = games[0].snapshot(0, PLAYERS[0])
    # out = games[0].snapshot(0, PLAYERS[2])
    # out = games[0].snapshot(1, PLAYERS[2])
    # out = games[0].snapshot(1, PLAYERS[0])
    out = games[0].snapshot(6, PLAYERS[0])
    return out


if __name__ == '__main__':
    test_snapshot()
