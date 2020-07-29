"""
This module holds classes that represent cards and their derivative classes.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Set


FACES = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', ]
FACES_ALT = {'j': 'J', 'q': 'Q', 'k': 'K', 'a': 'A'}

SUITS = ['♠', '♥', '♦', '♣', ]
SUITS_ALT = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣',
             's': '♠', 'h': '♥', 'd': '♦', 'c': '♣', }

class SuitType(Enum):
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

    def __init__(self):
        self._suit_type = TrumpType.NT

    @property
    def suit(self):
        return self._suit_type

    @suit.setter
    def suit(self, new_suit: TrumpType):
        self._suit_type = new_suit

Trump_singleton = Trump()
@dataclass
class Suit:
    suit_type: SuitType
    trump_suit: Trump = False

    @property
    def is_trump(self):
        return self.trump_suit.suit.value == self.suit_type.value

    def __repr__(self) -> str:
        return self.suit_type.value

    def __eq__(self, other):
        return self.suit_type.value == other.suit_type.value

    def __ne__(self, other):
        return self.suit_type.value != other.suit_type.value

    def __lt__(self, other):
        if self != other:
            if self.is_trump:
                return False

            if other.is_trump:
                return True

            return SUITS.index(self.suit_type.value) > SUITS.index(other.suit_type.value)

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

        :param face:
        :param suit:
        :raises ValueError: If `face` or `suit` are unsupported.
        """
        suit_type = SuitType.from_str(suit)
        self.suit = Suit(suit_type)
        self.is_trump = self.suit.is_trump
        if face.capitalize() not in FACES:
            raise ValueError(f"Unsupported Card Value {face}, must be one of {set(FACES)}")

        self.face = face.capitalize()

    def __repr__(self):
        return self.face + repr(self.suit)

    def __eq__(self, other):
        return self.face == other.face and self.suit == other.suit

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        # if self.face != other.face:
        #     return FACES.index(self.face) < FACES.index(other.face)
        # else:
        #     return SUITS.index(self.suit) > SUITS.index(other.suit)
        # return SUITS.index(self.suit) > SUITS.index(other.suit)
        if other.is_trump and not self.is_trump:
            return True

        if self.suit == other.suit:
            return FACES.index(self.face) < FACES.index(other.face)

        return False

    def __gt__(self, other):
        return other < self

    def __le__(self, other):
        return not (other < self)

    def __ge__(self, other):
        return not (self < other)

class Deck:  # [oriyan] Probably should be singelton

    def __init__(self):
        self.cards = []
        for face in FACES:
            for suit in SUITS:
                card = Card(face, suit)
                self.cards.append(card)

    def deal(self):
        shuffled_deck = np.random.permutation(self.cards).reshape(4,13)
        return shuffled_deck.tolist()

class Hand:
    def __init__(self, cards: List[Card]):
        # self.cards = set(cards)
        self.cards = sorted(cards, reverse=True)  # The sorting is needed for the agents!

    def play_card(self, card):
        self.cards.remove(card)

    def __str__(self):
        ret = ""
        for suit in SUITS:
            ret += f"{suit}:  "
            for card in self.cards:
                if card.suit == suit:
                    ret += f"{card.face} "
            ret += f"\n"

        return ret
