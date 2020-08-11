"""
This module holds classes that represent cards and their derivative classes.
"""

import numpy as np
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List


FACES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', ]

face_value = {'A': 4.5,
              'K': 3,
              'Q': 1.5,
              'J': 0.75,
              '10': 0.25,
              '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}


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
        if suit.upper() == 'NT':
            return TrumpType.NT

        if suit == 'NoTrump':
            return TrumpType.NoTrump
        try:
            suit_key = suit.capitalize()
            return TrumpType[suit_key]

        except KeyError:
            raise ValueError(f"Unsupported Suit {suit}. "
                             f"Must be one of {set(suit.name for suit in list(TrumpType))}")



@dataclass
class Suit:
    suit_type: SuitType
    trump_suit: TrumpType = TrumpType.NT

    @property
    def is_trump(self):
        return self.trump_suit.value == self.suit_type.value

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

    def __hash__(self) -> int:
        return hash(self.suit_type)

class Card:
    """
    A playing card.
    """

    def __init__(self, face: str, suit: str, trump: TrumpType = TrumpType.NT):
        """

        :param face: value of card - one of {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}

        :param suit: suit of card, one of {'S' or 'Spades', 'C' or 'Clubs', 'D' or 'Diamonds, 'H' or 'Hearts'}
        :raises ValueError: If `face` or `suit` are unsupported.
        """
        suit_type = SuitType.from_str(suit)
        self.suit = Suit(suit_type, trump_suit=trump)
        self.is_trump = self.suit.is_trump
        if face.capitalize() not in FACES:
            raise ValueError(
                f"Unsupported Card Value {face}, must be one of {set(FACES)}")

        self.face = face.capitalize()

    def __copy__(self):
        new_card = Card(self.face, self.suit.suit_type.name, self.suit.trump_suit)
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

    def __hash__(self) -> int:
        return hash((self.suit, self.face))

class Deck:
    """ Deck of cards."""

    def __init__(self, trump: TrumpType = TrumpType.NT):
        self.trump = trump
        self.cards = []
        for face in FACES:
            for suit in SUITS_ALT:
                card = Card(face, suit, self.trump)
                self.cards.append(card)

    def deal(self, cards_in_hand=13, recreate_game=''):
        """
        Returns 4 randomly dealt Hands, one for each player in the game.
        :param recreate_game: if supplied, will allow recreating a set of hands from a database. Currently unsupported.
        :returns List[Hand]: 4 hands
        """
        if not recreate_game:
            cards = np.random.choice(self.cards, 4*cards_in_hand, replace=False)
            shuffled_deck = \
                np.random.permutation(cards).reshape(4, cards_in_hand).tolist()
            hands = [Hand(cards) for cards in shuffled_deck]
            return hands
        # todo(oriyan/mar): create new deck from database representation


class Hand:
    """ A Player's hand . Holds their cards."""

    def __init__(self, cards: List[Card]):
        """ Initial hand of player is initialized with list of Card object."""
        self.cards = cards
        assert len(set(cards)) == len(cards)

    def __len__(self):
        return len(self.cards)

    def __copy__(self):
        cards = [copy(card) for card in self.cards]
        return Hand(cards)

    def play_card(self, card: Card):
        """ Plays card from hand. After playing this card, it is no longer available in the player's hand."""
        assert card in self.cards
        prev_num_cards = len(self.cards)
        self.cards.remove(card)
        assert len(self.cards) != prev_num_cards

    def get_cards_from_suite(self, suite: Suit, already_played):
        """ Returns all cards from player's hand that are from `suite`.
        If None, returns all cards."""
        if suite is None:
            cards = self.cards
            assert already_played.isdisjoint(cards)
            return self.cards

        cards = list(filter(lambda card: card.suit == suite, self.cards))
        assert already_played.isdisjoint(cards)
        return cards

    def get_cards_not_from_suite(self, suite: Suit):
        """ Returns all cards from player's hand that are not from `suite`.
                If None, returns all cards."""
        if suite is None:
            return self.cards

        cards = list(filter(lambda card: card.suit != suite, self.cards))
        return cards

    def get_cards_sorted_by_suits(self, already_played):
        sorted_hand = dict()
        trump = []
        for card in self.cards:
            if not sorted_hand.get(card.suit.suit_type):
                sorted_suit = sorted(self.get_cards_from_suite(card.suit, already_played))
                if card.is_trump:
                    trump = sorted_suit
                else:
                    sorted_hand[card.suit.suit_type] = sorted_suit
        return sorted_hand, trump

    def get_hand_value(self, already_played):
        """
        calculated following the steps:
        1. HCP value of the hand
        2. adjust of HCP
        3. adjust suit lenght
        4. adjust hands with top five cards of a suit
        5. total starting points (added in the heuristic)
        :param already_played:
        :return:
        """
        hand_value = 0
        adjust_suit_lenght = 0
        aces_10s_count = 0
        queens_jecks_count = 0

        hand_values_by_suits = dict()
        for card in self.cards:
            if not hand_values_by_suits.get(card.suit.suit_type):
                card_of_suit = self.get_cards_from_suite(card.suit, already_played)
                if len(card_of_suit) > 0:
                    adjust_suit_lenght += max([0, len(card_of_suit) - 4])
                values_of_card = [face_value[card.face] for card in card_of_suit
                                  if face_value[card.face] != 0]
                hand_values_by_suits[card.suit.suit_type] = values_of_card

        for suit, values in hand_values_by_suits.items():
            if len(values) > 0:
                hand_value += sum(values)
                aces_10s_count += values.count(0.25)  # tens
                aces_10s_count += values.count(4.5)  # aces
                queens_jecks_count += values.count(1.5)  # queens
                queens_jecks_count += values.count(0.75)  # jecks
                if len(values) == 5:  # adjust rule 4
                    hand_value += 3

        adjust_hand_value_value = abs(aces_10s_count - queens_jecks_count)
        sign = 1 if aces_10s_count > queens_jecks_count else -1
        if adjust_hand_value_value > 2:
            hand_value += sign
            if adjust_hand_value_value > 5:
                hand_value += sign

        return hand_value

    def __str__(self):
        ret = ""
        for suit in SUITS:
            ret += f"{suit}:  "
            for card in self.cards:
                if card.suit == suit:
                    ret += f"{card.face} "
            ret += f"\n"

        return ret
