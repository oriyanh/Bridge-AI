from copy import deepcopy
from itertools import cycle, islice
from typing import Tuple, List, TextIO

from cards import Card, SUITS, SuitType, TrumpType, Hand
from game import SimulatedGame
from multi_agents import IAgent
from players import Player, PositionEnum, POSITIONS, PLAYERS_CYCLE, TEAMS, Team
from state import State
from trick import Trick

TEAMS_CYCLE = {TEAMS[0]: TEAMS[1], TEAMS[1]: TEAMS[0]}
PLAYERS_DICT = {'N': 0, 'E': 1, 'S': 2, 'W': 3}


class TrickValidation(Trick):
    """
    Class of Trick, with added characteristic of order of cards given by
    remembering the starting player.
    """
    def __init__(self):
        super().__init__({})
        self.first_position = None

    def add_first_position(self, position: PositionEnum):
        self.first_position = position


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

    def position_to_player(self, position: PositionEnum):
        for p in self.players:
            if p.position == position:
                return p
        else:
            raise ValueError("something wrong in self.players")

    def snapshot(self, trick_idx: int, position: PositionEnum) -> \
            Tuple[List[Player], Trick, Card]:
        """
        Image of one moment in game, in trick_idx trick, when player should play
        :param trick_idx: first trick is 0
        :param position: the player to commit its turn now
        :return: current hands situation (ordered list of players), trick on
            desk and the chosen card (by player in given position).
        """
        if trick_idx >= len(self.tricks):
            raise IndexError(f"trick_idx argument has to be smaller then "
                             f"{len(self.tricks)}")

        # Load initial hands situation
        curr_hands = deepcopy(self.players)

        # Remove cards from all last tricks from hands
        for i in range(trick_idx):
            for j, p in enumerate(curr_hands):
                curr_hands[j].hand.play_card(self.tricks[i].trick[p])

        # Remove cards of current trick from hands of all players which play
        # before given player. In addition, store these cards
        curr_pos = self.tricks[trick_idx].first_position
        curr_player = self.position_to_player(curr_pos)
        curr_trick = Trick({})
        while curr_player.position != position:
            curr_hands[PLAYERS_DICT[curr_player.position.name]].play_card(
                self.tricks[trick_idx].trick[curr_player])
            curr_trick.add_card(
                curr_player, self.tricks[trick_idx].trick[curr_player])
            curr_player = self.players[PLAYERS_DICT[
                PLAYERS_CYCLE[curr_player.position].name]]
        chosen_card = self.tricks[trick_idx].trick[curr_player]

        return curr_hands, curr_trick, chosen_card

    def all_relevant_snapshots(self) -> \
            Tuple[List[List[Player]], List[Trick], List[Card]]:
        """
        Get all relevant data from DataGame: snapshots from all tricks, for the
            positions of both winners.
        :return: tuple of 3 elements:
            1. list of all sets-of-hands during game, one set-of-hands for every trick
            2. list of all open cards in specific trick, one set-of-cards for every trick
            3. list of all real cards the player should act, one for every trick
            all above are *concatenation* of results of both winners. For example,
            cards_list = [cards_list_winner_1] + [cards_list_winner_2]
        """
        winners_indices = [w.value - 1 for w in self.winner]
        # List of hands and trick for every winner
        hands_list_1: List[List[Player]] = []

        curr_hands = deepcopy(self.players)
        for trick_idx in range(len(self.tricks)):
            hands_list_1.append(deepcopy(curr_hands))
            for player_idx, player in enumerate(curr_hands):
                curr_hands[player_idx].hand.play_card(
                    self.tricks[trick_idx].trick[player])
        hands_list_2 = deepcopy(hands_list_1)

        trick_list_1: List[Trick] = []
        trick_list_2: List[Trick] = []
        trick_list = [trick_list_1, trick_list_2]
        chosen_cards_list_1: List[Card] = []
        chosen_cards_list_2: List[Card] = []
        chosen_cards_list = [chosen_cards_list_1, chosen_cards_list_2]

        for winner_idx, winner_list in enumerate([hands_list_1, hands_list_2]):
            for trick_idx, hands in enumerate(winner_list):
                curr_player = self.position_to_player(
                    self.tricks[trick_idx].first_position)
                curr_trick = Trick({})
                while curr_player.position != POSITIONS[winners_indices[winner_idx]]:
                    winner_list[trick_idx][
                        PLAYERS_DICT[curr_player.position.name]].play_card(
                        self.tricks[trick_idx].trick[curr_player])
                    curr_trick.add_card(
                        curr_player, self.tricks[trick_idx].trick[curr_player])
                    curr_player = self.players[PLAYERS_DICT[
                        PLAYERS_CYCLE[curr_player.position].name]]
                trick_list[winner_idx].append(curr_trick)
                chosen_cards_list[winner_idx].append(
                    self.tricks[trick_idx].trick[curr_player])

        return hands_list_1 + hands_list_2, trick_list_1 + trick_list_2, \
            chosen_cards_list_1 + chosen_cards_list_2


class Parser:
    def __init__(self, file_paths: List[str]):
        self.games: List[DataGame] = []
        for file in file_paths:
            self.games += (self.parse_file(file))

    def parse_file(self, file_name: str) -> List[DataGame]:
        """
        Get a path of PBN file, and parse it to list of DataGame objects
        """
        games = []
        with open(file_name) as f:
            for line in f:
                if line[1:6] == "Deal ":
                    players_line = line
                elif line[1:9] == "Declarer":
                    win_team, trump = self._parse_winners_and_trump(line, f)
                    if trump is None:
                        continue
                    players = self._parse_players(players_line[7:-3], trump)
                elif line[1:5] == "Play":
                    tricks = self._parse_tricks(line, f, players)
                    if len(tricks) == 0 or trump is None:  # delete empty games
                        continue
                    games.append(DataGame(players, tricks, win_team, trump))
        return games

    @staticmethod
    def _parse_players(line: str, trump: TrumpType) -> List[Player]:
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
                    cards.append(Card(face=face, suit=SuitType(SUITS[i]).name,
                                      trump=trump))
                    next_position = PLAYERS_CYCLE[curr_position]
            players[curr_position.value - 1] = Player(curr_position, Hand(cards))

        return players

    @staticmethod
    def _parse_winners_and_trump(line: str, f: TextIO) -> \
            Tuple[Tuple[PositionEnum], TrumpType]:
        """
        Helper for parse_file.
        Example: for sequential 3 lines [Declarer "E"], [Contract "4H"], [Result "10"]
            the result is that trump is ♥ and the team east-west is the winner.
        :param line: line from PBN file, which starts with "[Declarer"
        :param f: TextIO object, of the PBN file that read
        :return: tuple of Enums of winners, and the trump suit.
        """
        # Irregular situations
        if line[11] == '\"' or line[11] == '^':
            return None, None

        declarer = POSITIONS[PLAYERS_DICT[line[11]]]
        declare_team = TEAMS[0] if declarer in TEAMS[0] else TEAMS[1]

        bid_line = f.readline()
        obligation = int(bid_line[11]) + 6
        if (bid_line[12:-3] == "NT") or (bid_line[12:-4] == "NT"):
            trump = TrumpType.NT
        else:
            trump = TrumpType[bid_line[12]]

        res_line = f.readline()
        result = int(res_line[8:-2].split('\"')[1][-1])
        win_team = declare_team if result >= obligation else TEAMS_CYCLE[declare_team]
        return win_team, trump

    @staticmethod
    def _parse_tricks(player_line: str, f: TextIO, players: List[Player]) -> \
            List[TrickValidation]:
        """
        Helper for parse_file.
        Example: for sequential lines [Play "W"], HT H3 HA H6, C2 C3 C9 CJ, ..., *
            the meaning is that W starts the first trick with card 10♥, then N with
            3♥, E with A♥ and S with 6♥. The winner was E, so he starts the next
            trick with 9♣ and so on. (Order of cards in file is not the order of
            game! First player in every line is [Play <>], and the winner is
            inferred from line before)
        :param player_line: line from PBN file, which starts with "["Play"
        :param f: TextIO object, of the PBN file that read
        :param players: list of 4 Player objects, sorted by (N, E, S, W)
        :return: list of all tricks of a game
        """
        first_position = POSITIONS[PLAYERS_DICT[player_line[7]]]
        iter_num = islice(cycle([0, 1, 2, 3]), PLAYERS_DICT[player_line[7]], None)
        curr_player = players[next(iter_num)]
        tricks = []

        trick_line = f.readline()
        while trick_line[0] != '*':
            curr_trick = TrickValidation()
            cards = trick_line.split(' ')
            # Irregular situations
            if '-' in cards or '-\n' in cards or len(cards) != 4:
                break
            for c in cards:
                curr_trick.add_card(curr_player, Card(face=c[1], suit=c[0]))
                curr_player = players[next(iter_num)]
                curr_trick.add_first_position(first_position)
            tricks.append(curr_trick)
            first_position = curr_trick.get_winner()
            trick_line = f.readline()

        return tricks


def validate_agent_action(dg: DataGame,
                          trick_idx: int,
                          position: PositionEnum,
                          agent: IAgent) -> bool:
    curr_hands, curr_trick, chosen_card = dg.snapshot(trick_idx, position)

    teams = [Team(curr_hands[0], curr_hands[2]),
             Team(curr_hands[1], curr_hands[3])]

    curr_state = State(trick=curr_trick,
                       teams=teams,
                       players=curr_hands,
                       prev_tricks=dg.tricks[:trick_idx],
                       score=dict.fromkeys(teams),
                       curr_player=curr_hands[position.value - 1])

    sg = SimulatedGame(agent=agent,
                       other_agent=None,
                       verbose_mode=False,
                       state=curr_state)

    played_card = sg.play_single_move()
    print(f"Expected: {chosen_card}, Actual: {played_card}")

    return played_card == chosen_card


def validate_agent_per_data_game(agent: IAgent, dg: DataGame) -> \
        Tuple[int, int]:
    """
    Validate a agent by comparing its performances to data.
    :param agent: IAgent to check vs the data
    :param dg: DataGame object
    :return: first integer is number of comperes, and second is num of succeeds
    """
    all_hands, all_tricks, chosen_cards = dg.all_relevant_snapshots()
    tricks_num = len(all_hands) // 2
    succeeds = 0

    for pos_idx, position in enumerate(dg.winner):
        for trick_idx in range(tricks_num):
            curr_hands = all_hands[pos_idx * tricks_num + trick_idx]
            curr_trick = all_tricks[pos_idx * tricks_num + trick_idx]
            chosen_card = chosen_cards[pos_idx * tricks_num + trick_idx]
            # Create teams, such that first team is the winner
            teams = [Team(curr_hands[0], curr_hands[2]),
                     Team(curr_hands[1], curr_hands[3])]
            if curr_hands[0].position not in dg.winner:
                teams[0], teams[1] = teams[1], teams[0]

            curr_state = State(trick=curr_trick,
                               teams=teams,
                               players=curr_hands,
                               prev_tricks=dg.tricks[:trick_idx],
                               score=dict.fromkeys(teams),
                               curr_player=curr_hands[position.value - 1])

            sg = SimulatedGame(agent=agent,
                               other_agent=None,
                               verbose_mode=False,
                               state=curr_state)

            played_card = sg.play_single_move(get_card_only=True)
            if played_card == chosen_card:
                succeeds += 1

        return len(all_hands), succeeds
