import abc
from typing import Dict

from numpy.random import choice

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
        return choice(player.hand.cards)


class LowestFirstAgent(Agent):
    """
    Always picks the lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        return player.hand.cards[-1]


class HighestFirstAgent(Agent):
    """
    Always picks the highest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        return player.hand.cards[0]


class HardGreedyAgent(Agent):
    """
    If can beat current trick cards - picks highest value card available.
    If cannot - picks lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player, hands, trick):
        if len(trick) == 0:
            # Trick is empty - play best card.
            return player.hand.cards[0]

        elif player.hand.cards[0] > max(trick.cards()):
            # Can be best in current trick.
            return player.hand.cards[0]

        else:
            # Cannot win - play worst card.
            return player.hand.cards[-1]


def get_weakest_winner(best_trick_card, hands, player) -> Card:
    """

    :param best_trick_card:
    :param hands:
    :param player:
    :return:
    """
    return min(filter(lambda i: i > best_trick_card, hands[player].cards))


class SoftGreedyAgent(Agent):
    """
    If can beat current trick cards - picks the lowest value card available
    that can become the current best in trick.
    If cannot - picks lowest value card.
    """

    def __init__(self):
        super().__init__()

    def get_action(self, player: Player, players, trick):

        if len(trick) == 0:
            # Trick is empty - play worst card.
            return player.hand.cards[-1]

        best_trick_card = max(trick.cards())
        if player.hand.cards[0] > best_trick_card:
            # Can be best in current trick.
            return min(filter(lambda i: i > best_trick_card, player.hand.cards))
        else:
            # Cannot win - play worst card.
            return player.hand.cards[-1]


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

    def get_action(self, player: Player, hands: Dict[Player, Hand], trick: Trick) -> Card:
        pass


class MCTSAgent(Agent):

    def __init__(self):
        super().__init__()

    def get_action(self, player: Player, hands: Dict[Player, Hand], trick: Trick) -> Card:
        pass