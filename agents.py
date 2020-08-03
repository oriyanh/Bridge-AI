import abc
from typing import Dict

import numpy as np

from cards import Card, Hand
from players import Player
from trick import Trick


class Agent(abc.ABC):
    """
    Strategy-based agent to participate in the game.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, player: Player, hands: Dict[Player, Hand],
                   trick: Trick) -> Card:
        """
        Pick a card to play based on the environment and a programmed strategy.
        :param player: Current player
        :param hands: Current hands in play
        :param trick: Current trick in play
        :return: The card to play.
        """
        raise NotImplementedError


class RandomAgent(Agent):
    """
    Picks random card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        return np.random.choice(player.get_legal_actions(trick))


class LowestFirstAgent(Agent):
    """
    Always picks the lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        legal_cards = player.get_legal_actions(trick)
        return min(legal_cards)


class HighestFirstAgent(Agent):
    """
    Always picks the highest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        legal_cards = player.get_legal_actions(trick)
        return max(legal_cards)


class HardGreedyAgent(Agent):
    """
    If can beat current trick cards - picks highest value card available.
    If cannot - picks lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        legal_cards = player.get_legal_actions(trick)
        best_card = max(legal_cards)
        if len(trick) == 0:
            # Trick is empty - play best card.
            return max(legal_cards)

        if best_card > max(trick.cards()):
            # Can be best in current trick.
            return max(legal_cards)

        # Cannot win - play worst card.
        return min(legal_cards)


class SoftGreedyAgent(Agent):
    """
    If can beat current trick cards - picks the lowest value card available
    that can become the current best in trick.
    If cannot - picks lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player: Player, players, trick):
        legal_cards = player.get_legal_actions(trick)

        if len(trick) == 0:
            # Trick is empty - play worst card.
            return min(legal_cards)

        best_trick_card = max(trick.cards())
        if max(legal_cards) > best_trick_card:
            # Can be best in current trick.
            return min(
                filter(lambda i: i > best_trick_card, player.hand.cards))
        # Cannot win - play worst card.
        return min(legal_cards)


class HumanAgent(Agent):
    """
    Ask user for card, in format of <face><suit>.
    <face> can be entered as a number or a lowercase/uppercase letter.
    <suit> can be entered as an ASCII of suit or a lowercase/uppercase letter.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        while True:
            inp = input()
            if inp == '':
                print(f"<EnterKey> is not a valid card, try again")
                continue
            try:
                c = Card(inp[:-1], inp[-1])
                if c in hands[player].cards:
                    return c
                else:
                    print(f"{c} is not in your hand, try again")

            except ValueError or IndexError or TypeError:
                print(f"{inp} is not a valid card, try again")


class MinMaxAgent(Agent):  # with Alpha Beta pruning

    def __init__(self):
        super().__init__()

    def get_action(self, player: Player, hands: Dict[Player, Hand],
                   trick: Trick) -> Card:
        pass


class MCTSAgent(Agent):

    def __init__(self):
        super().__init__()

    def get_action(self, player: Player, hands: Dict[Player, Hand],
                   trick: Trick) -> Card:
        pass
