from copy import copy
from typing import Dict, KeysView, ValuesView, ItemsView

from cards import Suit, Card
from players import POSITIONS, Player, PositionEnum


class Trick:
    """
    A set of 4 cards played by each player in turn, during the play of a deal.
    """

    def __init__(self, trick, starting_suit=None):
        self.trick: Dict[Player, Card] = trick
        self.starting_suit = starting_suit  # type: Suit

    def __len__(self):
        return len(self.trick)

    def __copy__(self):
        trick = Trick({})
        for player, card in self.trick.items():
            trick.add_card(copy(player), card)
        return trick

    def create_from_other_players(self, players):
        new_trick = Trick({}, self.starting_suit)
        if self.trick is None:
            return new_trick
        for player in players:
            card = self.trick.get(player)
            if card:
                new_trick.add_card(player, card)
        return new_trick

    def players(self) -> KeysView[Player]:
        """
        Get all players with cards in current trick.
        :return: Iterable of all players.
        """
        return self.trick.keys()

    def cards(self) -> ValuesView[Card]:
        """
        Get all cards in current trick.
        :return: Iterable of all cards.
        """
        return self.trick.values()

    def items(self) -> ItemsView[Player, Card]:
        """
        Get all pairs of players and their cards in current trick.
        :return: Iterable of all pairs.
        """
        return self.trick.items()

    def add_card(self, player: Player, card: Card) -> None:
        """
        Add player's action to trick.
        :param player: The player placing the action.
        :param card: The action being played.
        :return: None
        """
        assert (player not in self.trick)
        if not self.trick:
            self.starting_suit = card.suit
        self.trick[player] = card

    def get_card(self, player: Player) -> Card:
        """
        Get the action that a player played.
        :param player: The player who's action in wanted.
        :return: The played action.
        """
        return self.trick.get(player)

    def get_winner(self) -> PositionEnum:
        """
        If all players played - return player with highest action.
        :return: Winning player.
        """
        assert (len(self.trick) == len(POSITIONS))
        relevant_players = []
        for player, card in self.items():
            if card.is_trump or card.suit == self.starting_suit:
                relevant_players.append(player)
        return max(relevant_players, key=self.trick.get)

    def reset(self) -> None:
        """
        Reset trick.
        :return: None
        """
        self.trick: Dict[Player, Card] = {}
        self.starting_suit = None
