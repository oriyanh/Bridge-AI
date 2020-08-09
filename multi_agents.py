import numpy as np

from abc import ABC, abstractmethod
from cards import Card
from state import State
from players import is_players_in_same_team, get_legal_actions, PLAYERS_CYCLE

FULL_TREE = -1


def lookup(name, namespace):
    """
    Get a method or class from any imported module from its name.
    Usage: lookup(functionName, globals())
    :returns: method/class reference
    :raises Exception: If the number of classes/methods existing in namespace with name is != 1
    """

    dots = name.count('.')
    if dots > 0:
        module_name, obj_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(module_name)
        return getattr(module, obj_name)
    else:
        modules = [obj for obj in namespace.values() if str(type(obj)) == "<type 'module'>"]
        options = [getattr(module, name) for module in modules if name in dir(module)]
        options += [obj[1] for obj in namespace.items() if obj[0] == name]
        if len(options) == 1:
            return options[0]
        if len(options) > 1:
            raise Exception('Name conflict for %s')
        raise Exception('%s not found as a method or class' % name)


class IAgent(ABC):
    """ Interface for bridge-playing agents."""

    @abstractmethod
    def __init__(self, target):
        self.target = target  # TODO [oriyan] Not completely sure what this should be -
        # it is only used in one place compared to a player's score.
        # Do we need this? Investigate later.

    @abstractmethod
    def get_action(self, state: State) -> Card:
        """
        Pick a action to play based on the environment and a programmed strategy.
        :param state:
        :return: The action to play.
        """
        raise NotImplementedError


# ------------------------------------SimpleAgent------------------------------------- #

class SimpleAgent(IAgent):
    """ Deterministic agent that plays according to input action."""

    def __init__(self, action_chooser_function='random_action', target=None):
        """
        :param str action_chooser_function: name of action to take .
        Should be a function [State] -> Card
        :param target: See comment in IAgent's constructor
        """
        self.action_chooser_function = lookup(action_chooser_function, globals())
        super().__init__(target)

    def get_action(self, state):
        return self.action_chooser_function(state)


def random_action(state):
    """
    Picks random action.
    :param State state:
    :returns Card: action to take
    """
    return np.random.choice(state.get_legal_actions())


def lowest_first_action(state):
    """
    Always picks the lowest value action.
    :param State state:
    :returns Card: action to take
    """
    return min(state.get_legal_actions())


def highest_first_action(state):
    """
    Always picks the highest value action
    :param State state:
    :returns Card: action to take
    """
    return max(state.get_legal_actions())


def hard_greedy_action(state):
    """
    If can beat current trick cards - picks highest value action available.
    If cannot - picks lowest value action.
    :param State state:
    :returns Card: action to take
    """
    legal_moves = state.get_legal_actions()
    best_move = max(legal_moves)
    if len(state.trick) == 0:  # Trick is empty - play best action.
        return best_move

    best_in_current_trick = max(state.trick.cards())
    worst_move = min(legal_moves)

    if best_move > best_in_current_trick:  # Can be best in current trick.
        return best_move
    else:  # Cannot win - play worst action.
        return worst_move


def hard_greedy_all_players_action(state):
    """
    If can beat current trick cards - picks highest value action available.
    If cannot - picks lowest value action.
    :param State state:
    :returns Card: action to take
    """
    legal_moves = state.get_legal_actions()

    best_move = max(legal_moves)
    if len(state.trick) == 0:  # Trick is empty - play best action.
        return starting_trick_card(state)

    best_in_current_trick = max(state.trick.cards())
    worst_move = min(legal_moves)

    op_cards = get_opponents_legal_card(state)
    if op_cards is not None:
        opponent_best = max(op_cards)
        card_to_win = max([opponent_best, best_in_current_trick])
        if best_move > card_to_win:
            return best_move

    elif best_move > best_in_current_trick:
        return best_move
    # Cannot win - play worst action.
    return worst_move


def soft_greedy_action(state):
    """
    If can beat current trick cards - picks the lowest value action available
    that can become the current best in trick.
    If cannot - picks lowest value action.
    :param State state:
    :returns Card: action to take
    """
    legal_moves = state.get_legal_actions()
    worst_move = min(legal_moves)
    best_move = max(legal_moves)

    if len(state.trick) == 0:  # Trick is empty - play worst action.
        return worst_move

    best_in_current_trick = max(state.trick.cards())

    if best_move > best_in_current_trick:  # Can be best in current trick.
        weakest_wining_move = min(filter(lambda move: move > best_in_current_trick, legal_moves))
        return weakest_wining_move
    return worst_move  # Cannot win - play worst action.


def soft_greedy_all_players_action(state):
    """
    If can beat current trick cards - picks the lowest value action available
    that can become the current best in trick.
    If cannot - picks lowest value action.
    :param State state:
    :returns Card: action to take
    """
    legal_moves = state.get_legal_actions()
    worst_move = min(legal_moves)
    best_move = max(legal_moves)

    if len(state.trick) == 0:  # Trick is empty - play worst action.
        return starting_trick_card(state)

    best_in_current_trick = max(state.trick.cards())

    op_cards = get_opponents_legal_card(state)
    if op_cards is not None:
        opponent_best = max(op_cards)
        card_to_win = max([opponent_best, best_in_current_trick])
        wining_moves = list(filter(lambda move: move > card_to_win, legal_moves))
        if len(wining_moves) > 0:
            return min(wining_moves)

    elif best_move > best_in_current_trick:
        weakest_wining_move = min(filter(lambda move: move > best_in_current_trick, legal_moves))
        return weakest_wining_move
    return worst_move  # Cannot win - play worst action.


def get_opponents_legal_card(state):
    i = len(state.trick)
    op_cards = None
    if i == 1 or i == 2:
        opponent = state.players_pos[PLAYERS_CYCLE[state.curr_player.position]]
        op_cards = get_legal_actions(state.trick.starting_suit, opponent)
    return op_cards


def starting_trick_card(state):
    curr_player_moves = state.get_legal_actions()

    opp1 = state.players_pos[PLAYERS_CYCLE[state.curr_player.position]]
    teammate = state.players_pos[PLAYERS_CYCLE[opp1.position]]
    opp2 = state.players_pos[PLAYERS_CYCLE[teammate.position]]

    # the following hands are sorted. 1st card is the smallest
    opp1_reg_cards, opp1_trump_cards = opp1.hand.get_cards_sorted_by_suits()
    opp2_reg_cards, opp2_trump_cards = opp2.hand.get_cards_sorted_by_suits()
    opp1_reg_cards.update(opp2_reg_cards)
    opp_reg_cards = opp1_reg_cards
    opp_trump_cards = opp1_trump_cards + opp2_trump_cards

    teammate_reg_cards, teammate_trump_cards = teammate.hand.get_cards_sorted_by_suits()

    cards = []
    for card in curr_player_moves:
        if opp_reg_cards.get(card.suit.suit_type) is not None:
            best_curr = card
            if teammate_reg_cards.get(card.suit):
                best_curr = max([teammate_reg_cards[card.suit.suit_type][-1], card])
            best_opp = max(opp_reg_cards[card.suit.suit_type][-1])
            if best_curr > best_opp:
                cards.append(card)
        else:
            if len(teammate_trump_cards) > 0:
                if len(opp_trump_cards) > 0:
                    best_curr = teammate_trump_cards[-1]
                    best_opp = opp_trump_cards[-1]
                    if best_curr > best_opp:
                        cards.append(card)
                else:
                    cards.append(card)
            else:
                if not len(opp_trump_cards) > 0:
                    cards.append(card)
    if len(cards) > 0:
        return np.random.choice(cards)
    return np.random.choice(curr_player_moves)

# ---------------------------MultiAgentSearchAgent--------------------------- #
class MultiAgentSearchAgent(IAgent):
    """Abstract agent implementing IAgent that searches a game tree"""

    def __init__(self, evaluation_function='score_evaluation_function',
                 depth=2, target=None):
        """
        :param evaluation_function: function mapping (State, *args) -> Card ,
            where *args is determined by the agent itself.
        :param int depth: -1 for full tree, any other number > 1 for depth bounded tree
        :param target:
        """
        self.evaluation_function = lookup(evaluation_function, globals())
        self.depth = depth
        super().__init__(target)

    def get_action(self, state):
        return NotImplementedError


class AlphaBetaAgent(MultiAgentSearchAgent):
    """ Agent implementing AlphaBeta pruning (with MinMax tree search)"""

    def __init__(self, evaluation_function='count_tricks_won_evaluation_function',
                 depth=2, target=None):
        super().__init__(evaluation_function, depth, target)

    def get_action(self, state):
        legal_moves = state.get_legal_actions()
        successors = [state.get_successor(action=action)
                      for action in legal_moves]

        if self.depth == 0:
            scores = [self.evaluation_function(successor)
                      for successor in successors]  # TODO [oriyan] Maryna,
            best_score = max(scores)
            best_indices = [index for index in range(len(scores))
                            if scores[index] == best_score]
            chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
            return legal_moves[chosen_index]
        else:
            a, b = -np.inf, np.inf
            chosen_index = 0
            for i, successor in enumerate(successors):
                next_child_score = self.score(successor, self.depth,
                                              1, False, a, b)
                if next_child_score > a:
                    chosen_index = i
                    a = next_child_score
                if b <= a:
                    break
            return legal_moves[chosen_index]

    def score(self, state, max_depth, curr_depth, is_max, a, b):
        """ Recursive method returning score for current state (the node in search tree).

        :param State state: State of game
        :param int max_depth: Max tree depth to search
        :param int curr_depth: Current depth in search tree
        :param bool is_max: True if current player is Max player, False if Min player
        :param float a: Current alpha score
        :param float b: Current beta score
        :returns float: Score for current state (the node)
        """
        if curr_depth == max_depth:
            return self.evaluation_function(state, is_max, self.target)

        # get current player moves
        curr_player = state.curr_player
        legal_moves = state.get_legal_actions()

        if not legal_moves:
            return self.evaluation_function(state, is_max, self.target)
        possible_states = [state.get_successor(action=action)
                           for action in legal_moves]

        if is_max:
            for next_state in possible_states:
                next_player = next_state.curr_player
                is_next_max = True if is_players_in_same_team(curr_player, next_player) else False
                next_depth = curr_depth if is_next_max else curr_depth + 1
                a = max((a, self.score(next_state, max_depth, next_depth,
                                       is_next_max, a, b)))
                if b <= a:
                    break
            return a

        for next_state in possible_states:
            next_player = next_state.curr_player
            is_next_max = False if is_players_in_same_team(curr_player, next_player) else True
            next_depth = curr_depth + 1 if is_next_max else curr_depth
            b = min((b, self.score(next_state, max_depth,
                                   next_depth, is_next_max, a, b)))
            if b <= a:
                break
        return b


def is_target_reached_evaluation_function(state, is_max=True, target=None):
    """
    Score of state is 1 if current player is Max player and target has been reached.
    0 Otherwise.

    :param State state: game state
    :param bool is_max: is max player
    :param target:
    :returns float: score of state
    """
    if not target:
        return 0
    # if max return True if met the score of the team of the current player
    # else return if the opposite team met the required score
    player_score = state.get_score(is_max)
    if target <= player_score:
        return 1
    return 0


def count_tricks_won_evaluation_function(state, is_max=True, target=None):
    """
    :param State state: game state
    :param bool is_max: is max player
    :param target:
    :returns float: score of state
    """
    return state.get_score(is_max)


def greedy_evaluation_function(state, is_max=True, target=None):
    """
    :param State state: game state
    :param bool is_max: is max player
    :param target:
    :returns float: score of state
    """
    value = state.get_score(is_max)
    if len(state.trick) == 0:  # Trick is empty - play worst action.
        return value

    soft_moves_count = soft_greedy_legal_moves_count(state)
    return 10 * value + soft_moves_count


def soft_greedy_legal_moves_count(state, ):
    legal_moves = state.get_legal_actions()
    best_move = max(legal_moves)
    best_in_current_trick = max(state.trick.cards())
    if best_move > best_in_current_trick:  # Can be best in current trick.
        count_wining_moves = len(list(filter(lambda move: move > best_in_current_trick,
                                             legal_moves)))
        return count_wining_moves
    return -1


# ---------------------------------MCTSAgent--------------------------------- #

# class MCTSAgent(IAgent):
#     def __init__(self, action_chooser_function='random_action', MCTSNode):
#         self.action_chooser_function = lookup(action_chooser_function,
#                                                    globals())
#         self.root = MCTSNode
#         super().__init__(action_chooser_function, MCTSNode)
#
#     def get_action(self, state, simulation_count=100):
#         for i in range(simulation_count):
#             leaf = self.traverse(self.root)
#             reward = leaf.rollout()
#             leaf.backpropagate(reward)
#             # to select best child go for exploitation only
#         return self.root.best_child(c_param=0.)
#
#     def traverse(self, node) -> MCTSNode:
#         while fully_expanded(node):
#             node = best_uct(node)
#
#             # in case no children are present / node is terminal
#         return pick_unvisited(node.children) or node


# ---------------------------------HumanAgent-------------------------------- #
class HumanAgent(IAgent):
    """
    Ask user for action, in format of <face><suit>.
    <face> can be entered as a number or a lowercase/uppercase letter.
    <suit> can be entered as an ASCII of suit or a lowercase/uppercase letter.
    """

    def __init__(self):
        super().__init__(self)

    def get_action(self, state):
        while True:
            inp = input()
            if inp == '':
                print(f"<EnterKey> is not a valid action, try again")
                continue
            try:
                card_suit, card_number = inp[:-1], inp[-1]
                action = Card(card_number, card_suit)
                legal_moves = state.get_legal_actions()
                if action in legal_moves:
                    return action
                else:
                    print(f"{card_suit, card_number} "
                          f"is not in your hand, try again")

            except ValueError or IndexError or TypeError:
                print(f"{inp} is not a valid action, try again")
