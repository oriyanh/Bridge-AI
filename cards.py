"""
This module holds classes that represent cards and their derivative classes.
"""

import numpy as np
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List


FACES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', ]

SUITS = ['♠', '♥', '♦', '♣', ]
SUITS_ALT = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}


class SuitType(Enum):
    """ Enum representing card suit"""

    Spades = '♠'
    S = '♠'

    Hearts = '♥'
    H = '♥'

    Diamonds = '♦'
    D = '♦'

    Clubs = '♣'
    C = '♣'

    @staticmethod
    def from_str(suit: str):
        """ Parses string into SuitType object

        :param suit: Suit string to parse into SuitType
        :returns SuitType: parsed suit
        :raises ValueError: If `suit` is unsupported.
        """

        try:
            suit_key = suit.capitalize()
            return SuitType[suit_key]

        except KeyError:
            raise ValueError(f"Unsupported Suit {suit}. "
                             f"Must be one of {set(suit.name for suit in list(SuitType))}")


class TrumpType(Enum):
    """ Enum representing match's trump suit"""

    Spades = '♠'
    S = '♠'

    Hearts = '♥'
    H = '♥'

    Diamonds = '♦'
    D = '♦'

    Clubs = '♣'
    C = '♣'

    NoTrump = 'NT'
    NT = 'NT'

    @staticmethod
    def from_str(suit: str):
        """ Parses string into SuitType object

        :param suit: Suit string to parse into SuitType
        :returns SuitType: parsed suit
        :raises ValueError: If `suit` is unsupported.
        """

        try:
            suit_key = suit.capitalize()
            return SuitType[suit_key]

        except KeyError:
            raise ValueError(f"Unsupported Suit {suit}. "
                             f"Must be one of {set(suit.name for suit in list(SuitType))}")


class Trump:
    """ Class representing the trump in the current game. Initialized as NT (No Trump)"""

    def __init__(self):
        self._suit_type = TrumpType.NT

    @property
    def suit(self):
        return self._suit_type

    @suit.setter
    def suit(self, new_suit: TrumpType):
        self._suit_type = new_suit


trump_singleton = Trump()


@dataclass
class Suit:
    suit_type: SuitType
    trump_suit: Trump = trump_singleton  # TODO [oriyan] need to take trump into consideration in each game, and set it accordingly.

    @property
    def is_trump(self):
        return self.trump_suit.suit.value == self.suit_type.value

    def __repr__(self) -> str:
        return self.suit_type.value

    def __eq__(self, other):
        if isinstance(other, str):
            return self.suit_type.value == other
        return self.suit_type.value == other.suit_type.value

    def __ne__(self, other):
        return self.suit_type.value != other.suit_type.value

    def __lt__(self, other):
        if self != other:
            if self.is_trump:
                return False

            if other.is_trump:
                return True

            return SUITS.index(self.suit_type.value) > SUITS.index(
                other.suit_type.value)

        return False

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return other <= self


class Card:
    """
    A playing card.
    """

    def __init__(self, face: str, suit: str):
        """

        :param face: value of card - one of {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}

        :param suit: suit of card, one of {'S' or 'Spades', 'C' or 'Clubs', 'D' or 'Diamonds, 'H' or 'Hearts'}
        :raises ValueError: If `face` or `suit` are unsupported.
        """
        suit_type = SuitType.from_str(suit)
        self.suit = Suit(suit_type)
        self.is_trump = self.suit.is_trump
        if face.capitalize() not in FACES:
            raise ValueError(
                f"Unsupported Card Value {face}, must be one of {set(FACES)}")

        self.face = face.capitalize()

    def __copy__(self):
        new_card = Card(self.face, self.suit.suit_type.name)
        new_card.is_trump = self.is_trump
        return new_card

    def __repr__(self):
        return f"{self.face}{self.suit}"

    def __eq__(self, other):
        return self.face == other.face and self.suit == other.suit

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        if other.is_trump and not self.is_trump:
            return True

        if self.is_trump and not other.is_trump:
            return False

        if self.suit == other.suit:
            return FACES.index(self.face) < FACES.index(other.face)

        return SUITS.index(self.suit.suit_type.value) < SUITS.index(
            self.suit.suit_type.value) and \
               FACES.index(self.face) < FACES.index(other.face)

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return not (other < self)

    def __ge__(self, other):
        return not (self < other)


class Deck:
    """ Deck of cards."""

    def __init__(self):
        self.cards = []
        for face in FACES:
            for suit in SUITS_ALT:
                card = Card(face, suit)
                self.cards.append(card)

    def deal(self, recreate_game=''):
        """
        Returns 4 randomly dealt Hands, one for each player in the game.
        :param recreate_game: if supplied, will allow recreating a set of hands from a database. Currently unsupported.
        :returns List[Hand]: 4 hands
        """
        if not recreate_game:
            shuffled_deck = \
                np.random.permutation(self.cards).reshape(4, 13).tolist()
            hands = [Hand(cards) for cards in shuffled_deck]
            return hands
        # todo(oriyan/mar): create new deck from database representation


class Hand:
    """ A Player's hand . Holds their cards."""

    def __init__(self, cards: List[Card]):
        """ Initial hand of player is initialized with list of Card object."""
        self.cards = cards

    def __len__(self):
        return len(self.cards)

    def __copy__(self):
        cards = [copy(card) for card in self.cards]
        return Hand(cards)

    def play_card(self, card: Card):
        """ Plays card from hand. After playing this card, it is no longer available in the player's hand."""
        self.cards.remove(card)

    def get_cards_from_suite(self, suite: Suit):
        """ Returns all cards from player's hand that are from `suite`.
        If None, returns all cards."""
        if suite is None:
            return self.cards

        cards = list(filter(lambda card: card.suit == suite, self.cards))
        return cards

    def __str__(self):
        ret = ""
        for suit in SUITS:
            ret += f"{suit}:  "
            for card in self.cards:
                if card.suit == suit:
                    ret += f"{card.face} "
            ret += f"\n"

        return ret
