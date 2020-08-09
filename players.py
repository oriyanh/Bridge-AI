from copy import copy
from enum import Enum
from typing import List

from cards import Hand, Card


PositionEnum = Enum("PlayersEnum", ['N', 'E', 'S', 'W'])

POSITIONS = list(PositionEnum)

TEAMS = [(PositionEnum.N, PositionEnum.S), (PositionEnum.W, PositionEnum.E)]

PLAYERS_CYCLE = {PositionEnum.N: PositionEnum.E,
                 PositionEnum.E: PositionEnum.S,
                 PositionEnum.S: PositionEnum.W,
                 PositionEnum.W: PositionEnum.N}

TEAMMATES = {PositionEnum.N: PositionEnum.S,
             PositionEnum.S: PositionEnum.N,
             PositionEnum.E: PositionEnum.W,
             PositionEnum.W: PositionEnum.E}


class Player:
    """ Represents one of the 4 players in the game."""

    def __init__(self, position: PositionEnum, hand: Hand):
        """

        :param position: Position of player
        :param hand: Initial hand of player
        """
        self.position = position
        self.hand = hand
        self.played = set()

    def __copy__(self):
        hand = copy(self.hand)
        player = Player(self.position, hand)
        player.played = set(self.played)
        return player

    def play_card(self, card: Card) -> None:
        """ Plays card from hand. card is no longer available."""
        assert card not in self.played
        self.hand.play_card(card)
        self.played.add(card)

    def get_legal_actions(self, trick, already_played) -> List[Card]:
        """ Returns list of legal actions for player in current trick

        :param Trick trick: Current trick
        :param already_played: Set of cards already used in state, used for unit testing.
        :returns: legal actions for player:
        """
        legal_actions = self.hand.get_cards_from_suite(trick.starting_suit, already_played)
        assert self.played.isdisjoint(legal_actions)
        assert already_played.isdisjoint(legal_actions)
        if not legal_actions:
            legal_actions = self.hand.cards
            assert already_played.isdisjoint(legal_actions)
        return legal_actions

    def __str__(self):
        return self.position.name

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


def get_legal_actions(suit, player, already_played) -> List[Card]:
    legal_actions = player.hand.get_cards_from_suite(suit, already_played)
    if not legal_actions:
        legal_actions = player.hand.cards
    else:
        trump_cards = [card for card in player.hand.cards if card.is_trump]
        legal_actions.extend(trump_cards)
    return legal_actions


class Team:
    """ Team of two players sitting on opposite sides of table."""

    def __init__(self, p0: Player, p1: Player):
        self.players = [p0, p1]
        self.teammate = {p0.position: p1,  # todo [ORIYAN] Possibly remove?
                         p1.position: p0}
        # todo(maryna): maybe add the score directly into the team object?

    def __copy__(self):
        p0, p1 = self.players[0], self.players[1]
        copy_p0, copy_p1 = copy(p0), copy(p1)
        return Team(copy_p0, copy_p1)

    def __str__(self):
        return f"{self.players[0]}{self.players[1]}"

    def has_player(self, p: Player) -> bool:
        """ Is player `p` on the team """
        return p in self.players

    def get_players(self) -> List[Player]:
        return self.players

    # more useful to be outside as static method.
    def get_teammate(self, p: Player) -> Player:
        assert (p in self.players)
        return self.teammate[p.position]

    def __hash__(self) -> int:
        return hash(frozenset(self.players))

    def __eq__(self, other) -> bool:
        return frozenset(self.teammate.keys()).issubset(other.teammate.keys())


def is_players_in_same_team(p1: Player, p2: Player) -> bool:
    if (p1.position, p2.position) in TEAMS or (p2.position, p1.position) in TEAMS:
        return True
    return False
