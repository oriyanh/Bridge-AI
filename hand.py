from card import SUITS
from players import Player


class Hand:
    def __init__(self, player: Player, cards: list):
        self.player = player
        self.cards = cards

        self.cards.sort(reverse=True)  # The sorting is needed for the agents!

    def __str__(self):
        ret = ""
        for suit in SUITS:
            ret += f"{suit}:  "
            for card in self.cards:
                if card.suit == suit:
                    ret += f"{card.face} "
            ret += f"\n"

        return ret
