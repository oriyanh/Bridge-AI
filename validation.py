from cards import Card, SUITS, SuitType, TrumpType, Hand, Deck
from players import Player, PositionEnum, POSITIONS, PLAYERS_CYCLE, TEAMS
from trick import Trick
from typing import Tuple, List, TextIO
from copy import copy
from itertools import cycle, islice


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

    def add_first_player(self, player: PositionEnum):
        self.first_player = player


class DataGame:
    """
    Class which stores a recorded game, with option to take a snapshot from any
    point in it.
    """
    def __init__(self, players: List[Player], tricks: List[TrickValidation],
                 winner: Tuple[PositionEnum], trump: TrumpType):
        self.players = players
        self.tricks = tricks
        self.winner = winner
        self.trump = trump

    def snapshot(self, trick_idx: int, player: PositionEnum):
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
        curr_hands = copy(self.players)

        # Remove cards from all last tricks from hands
        for i in range(trick_idx):
            for j, p in enumerate(curr_hands):
                curr_hands[j].hand.play_card(self.tricks[i].trick[p])

        # Remove cards of current trick from hands of all players which play
        # before given player. In addition, store these cards
        curr_player = self.tricks[trick_idx].first_player  # todo: first player Enum
        curr_trick = Trick({})
        while curr_player != player:
            curr_hands[PLAYERS_DICT[curr_player]].play_card(
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
                tricks = parse_tricks(line, f, players)
                games.append(DataGame(players, tricks, win_team, trump))
    return games


def parse_players(line: str) -> List[Player]:
    """
    Helper for parse_file.
    Example: line is such -
        [Deal "E:AK872.KQJT.K.Q94 QT95.85.AQJ2.AK7 4.A962.96.J86532 J63.743.T87543.T"]
        And the result is list of Player object. First is Player(PositionEnum.E, Hand)
        such that Hand is list contain A, K, 8, 7, 2 faces of suit ♠, ect.
    :param line: line from PBN file, which starts with "[Deal "
    :return: list of 4 Player objects, sorted by (N, E, S, W)
    """
    player_str, all_hands_str = line.split(':')
    next_position = POSITIONS[PLAYERS_DICT[player_str]]

    players = [None, None, None, None]
    players_str = all_hands_str.split(' ')  # spaces separate every two players
    for p in players_str:
        curr_position = next_position
        cards_str = p.split('.')  # dots separate every two suits
        cards = []
        for i, suit in enumerate(cards_str):
            for face in suit:
                cards.append(Card(face=face, suit=SuitType(SUITS[i]).name))
                next_position = PLAYERS_CYCLE[curr_position]
        players[curr_position.value - 1] = Player(curr_position, Hand(cards))

    return players


def parse_winners_and_trump(line: str, f: TextIO) -> Tuple[Tuple[PositionEnum],
                                                           TrumpType]:
    """
    Helper for parse_file.
    Example: for sequential 3 lines [Declarer "E"], [Contract "4H"], [Result "10"]
        the result is that trump is ♥ and the team east-west is the winner.
    :param line: line from PBN file, which starts with "[Declarer"
    :param f: TextIO object, of the PBN file that read
    :return: tuple of Enums of winners, and the trump suit.
    """
    declarer = POSITIONS[PLAYERS_DICT[line[11]]]
    declare_team = TEAMS[0] if declarer in TEAMS[0] else TEAMS[1]

    bid_line = f.readline()
    obligation = int(bid_line[11]) + 6
    trump = TrumpType.NT if bid_line[12:-3] == "NT" else TrumpType[bid_line[12]]

    res_line = f.readline()
    result = int(res_line[8:-2].split('\"')[1])
    win_team = declare_team if result >= obligation else TEAMS_CYCLE[declare_team]
    return win_team, trump


def parse_tricks(player_line: str, f: TextIO, players: List[Player]) -> \
        List[TrickValidation]:
    """
    Helper for parse_file.
    Example: for sequential lines [Play "S"], CA C2 CT C4, H8 H2 H4 HJ, ..., *
        the meaning is that S starts the first trick with card 10♣, then W with
        4♣, N with A♣ and E with 2♣. The winner was N, so he starts the next
        trick with 8♥ and so on. (Order of cards in file is not the order of
        game! N is written first, and the winner is inferred from line before)
    :param player_line: line from PBN file, which starts with "["Play"
    :param f: TextIO object, of the PBN file that read
    :param players: list of 4 Player objects, sorted by (N, E, S, W)
    :return: list of all tricks of a game
    """
    first_position = POSITIONS[PLAYERS_DICT[player_line[7]]]
    iter_num = islice(cycle([0, 1, 2, 3]), 0, None)
    curr_player = players[next(iter_num)]
    tricks = []

    trick_line = f.readline()
    while trick_line[0] != '*':
        curr_trick = TrickValidation()
        cards = trick_line.split(' ')
        if '-' in cards:  # last trick in game, some players give up
            break
        for c in cards:
            curr_trick.add_card(curr_player, Card(face=c[1], suit=c[0]))
            curr_player = players[next(iter_num)]
            curr_trick.add_first_player(first_position)
        tricks.append(curr_trick)
        first_position = curr_trick.get_winner()
        trick_line = f.readline()

    return tricks


def test_snapshot():
    file = "bridge_data_1/Aulpll23.pbn"
    # file = "bridge_data_1/Dkofro20.pbn"
    games = parse_file(file)
    # out = games[0].snapshot(0, POSITIONS[0])
    # out = games[0].snapshot(0, POSITIONS[2])
    # out = games[0].snapshot(1, POSITIONS[2])
    out = games[0].snapshot(1, POSITIONS[0])
    # out = games[0].snapshot(6, POSITIONS[0])
    return out


if __name__ == '__main__':
    test_snapshot()
