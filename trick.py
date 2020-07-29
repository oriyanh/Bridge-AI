from typing import Dict, KeysView, ValuesView, ItemsView

from card import Card
from players import Player, PLAYERS


class Trick:
    """
    A set of 4 cards played by each player in turn, during the play of a deal.
    """
    def __init__(self):
        self.trick: Dict[Player, Card] = {}

    def __len__(self):
        return len(self.trick)

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
        Add player's card to trick.
        :param player: The player placing the card.
        :param card: The card being played.
        :return: None
        """
        assert (player not in self.trick)
        self.trick[player] = card

    def get_card(self, player: Player) -> Card:
        """
        Get the card that a player played.
        :param player: The player who's card in wanted.
        :return: The played card.
        """
        return self.trick[player] if player in self.trick else None

    def get_winner(self, current_suit=None, trump=None) -> Player:
        """
        If all players played - return player with highest card.
        :return: Winning player.
        """
        assert (len(self.trick) == len(PLAYERS))
        return max(self.trick, key=self.trick.get)

    def reset(self) -> None:
        """
        Reset trick.
        :return: None
        """
        self.trick: Dict[Player, Card] = {}
