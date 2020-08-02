import numpy as np

import util
from abc import ABC, abstractmethod
from cards import Card
from state import State

FULL_TREE = -1


class IAgent(ABC):
    @abstractmethod
    def __init__(self, target):
        self.target = target

    @abstractmethod
    def get_action(self, state: State) -> Card:
        """
        Pick a action to play based on the environment and a programmed strategy.
        :param state:
        :return: The action to play.
        """
        raise NotImplementedError


# ------------------------------------MultiAgentSingleAction------------------------------------- #

class SingleActionAgent(IAgent):
    def __init__(self, action_chooser_function='random_action', target=None):
        self.action_chooser_function = util.lookup(action_chooser_function, globals())
        super().__init__(target)

    def get_action(self, state):
        return self.action_chooser_function(state)


def random_action(state):
    """
    Picks random action.
    :param state:
    :return:
    """
    return np.random.choice(state.get_legal_actions())


def lowest_first_action(state):
    """
    Always picks the lowest value action.
    :param state:
    :return:
    """
    return min(state.get_legal_actions())


def highest_first_action(state):
    """
    Always picks the highest value action
    :param state:
    :return:
    """
    return max(state.get_legal_actions())


def hard_greedy_action(state):
    """
    If can beat current trick cards - picks highest value action available.
    If cannot - picks lowest value action.
    :param state:
    :return:
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


def soft_greedy_action(state):
    """
    If can beat current trick cards - picks the lowest value action available
    that can become the current best in trick.
    If cannot - picks lowest value action.
    :param state:
    :return:
    """
    legal_moves = state.get_legal_actions()
    worst_move = min(legal_moves)
    best_move = max(legal_moves)

    if len(state.trick) == 0:  # Trick is empty - play worst action.
        return worst_move

    best_in_current_trick = max(state.trick.cards())
    # todo could be a fixed bug compered to git

    if best_move > best_in_current_trick:  # Can be best in current trick.
        weakest_wining_move = min(filter(lambda move: move > best_in_current_trick, legal_moves))
        return weakest_wining_move
    return worst_move  # Cannot win - play worst action.

# ------------------------------------MultiAgentSearchAgent------------------------------------- #


class MultiAgentSearchAgent(IAgent):
    def __init__(self, evaluation_function='score_evaluation_function', depth=2, target=None):
        """
        :param evaluation_function:
        :param depth: -1 for full tree, any other number > 1 for depth bounded tree
        """
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth
        super().__init__(target)

    def get_action(self, state):
        pass


class AlphaBetaAgent(MultiAgentSearchAgent):
    def __init__(self, evaluation_function='count_tricks_won_evaluation_function', depth=2, target=None):
        super().__init__(evaluation_function, depth, target)

    def get_action(self, state):
        legal_moves = state.get_legal_actions()
        successors = [state.get_successors(action=action) for action in legal_moves]

        if self.depth == 0:
            scores = [self.evaluation_function(successor) for successor in successors]
            best_score = max(scores)
            best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
            chosen_index = np.random.choice(best_indices)  # Pick randomly among the best
            return legal_moves[chosen_index]
        else:
            a, b = -np.inf, np.inf
            chosen_index = 0
            for i, successor in enumerate(successors):
                next_child_score = self.score(successor, self.depth, 1, False, a, b)
                if next_child_score > a:
                    chosen_index = i
                    a = next_child_score
                if b <= a:
                    break
            # print(f"{legal_moves}, count: {len(legal_moves)}, index: {chosen_index}")
            return legal_moves[chosen_index]

    def score(self, state, max_depth, curr_depth, is_max, a, b):
        """
        a - the best value for max, b - the best value for min
        """
        if curr_depth == max_depth:
            return self.evaluation_function(state, is_max, self.target)

        # get current player moves
        current_player = state.curr_player
        legal_moves = state.get_legal_actions()

        if not legal_moves:
            return self.evaluation_function(state, is_max, self.target)
        possible_states = [state.get_successors(action=action) for action in legal_moves]

        if is_max:
            for next_state in possible_states:
                next_player = next_state.curr_player
                is_next_max = True if next_player == current_player else False
                next_depth = curr_depth if is_next_max else curr_depth + 1
                a = max((a, self.score(next_state, max_depth, next_depth, is_next_max, a, b)))
                if b <= a:
                    break
            return a

        for next_state in possible_states:
            next_player = next_state.curr_player
            is_next_max = True if next_player != current_player else False
            next_depth = curr_depth + 1 if is_next_max else curr_depth
            b = min((b, self.score(next_state, max_depth, next_depth, is_next_max, a, b)))
            if b <= a:
                break
        return b


def is_target_reached_evaluation_function(state, is_max=True, target=None):
    if not target:
        return 0
    # if max return True if met the score of the team of the current player
    # else return if the opposite team met the required score
    player_score = state.get_score(is_max)
    if target <= player_score:
        return 1
    return 0


def count_tricks_won_evaluation_function(state, is_max=True, target=None):
    return state.get_score(is_max)

# ------------------------------------------MTCSAgent------------------------------------------- #

#
# class MCTSAgent(IAgent):
#     def __init__(self, action_chooser_function='random_action', MCTSNode):
#         self.action_chooser_function = util.lookup(action_chooser_function, globals())
#         self.root = MCTSNode
#         super().__init__(action_chooser_function, MCTSNode)
#
#     def get_action(self, state, simulation_count=100):
#         for i in range(simulation_count):
#             leaf = self.traverse(self.root)
#             reward = leaf.rollout()
#             leaf.backpropagate(reward)  # to select best child go for exploitation only
#         return self.root.best_child(c_param=0.)
#
#     def traverse(self, node) -> MCTSNode:
#         while fully_expanded(node):
#             node = best_uct(node)
#
#             # in case no children are present / node is terminal
#         return pick_univisted(node.children) or node


# -------------------------------------------HumanAgent----------------------------------------- #


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
                    print(f"{card_suit, card_number} is not in your hand, try again")

            except ValueError or IndexError or TypeError:
                print(f"{inp} is not a valid action, try again")
