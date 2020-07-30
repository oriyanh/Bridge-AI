from typing import Dict, KeysView, ValuesView, ItemsView

from cards import Suit, Card
from players import Player, POSITIONS, Player, PositionEnum


class Trick:
    """
    A set of 4 cards played by each player in turn, during the play of a deal.
    """
    def __init__(self):
        self.trick: Dict[Player, Card] = {}
        self.starting_suit = None  #type: Suit

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
        if not self.trick:
            self.starting_suit = card.suit
        self.trick[player] = card

    def get_card(self, player: Player) -> Card:
        """
        Get the card that a player played.
        :param player: The player who's card in wanted.
        :return: The played card.
        """
        return self.trick.get(player)

    def get_winner(self) -> PositionEnum:
        """
        If all players played - return player with highest card.
        :return: Winning player.
        """
        assert (len(self.trick) == len(POSITIONS))
        relevant_players = []
        for player, card in self.items():
            if card.is_trump or \
                card.suit == self.starting_suit:
                relevant_players.append(player)
        return (max(relevant_players, key=self.trick.get)).position

    def reset(self) -> None:
        """
        Reset trick.
        :return: None
        """
        self.trick: Dict[Player, Card] = {}
        self.starting_suit = None
