import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import copy
from queue import Queue
from typing import Dict, List, Set
from concurrent.futures.thread import ThreadPoolExecutor

from cards import Card
from game import SimulatedGame
from state import State
from players import is_players_in_same_team



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

        :param str action_chooser_function: name of action to take, or a function.
            Function should map State -> Card
        :param target: See comment in IAgent's constructor
        """

        if isinstance(action_chooser_function, str):
            self.action_chooser_function = lookup(action_chooser_function, globals())
        else:
            self.action_chooser_function = action_chooser_function
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


def add_randomness_to_action(func, epsilon):
    """
    Wraps a `State->Card` function with a randomizing factor -
        w.p. epsilon, action is chosen at random.
    :param func: `State->Card` function
    :param float epsilon: Probability of choosing action at random. In range [0,1]
    :returns: Function mapping `State->Card` with additional randomizing factor
    """

    def randomized_action(state):
        if np.random.rand() < epsilon:
            return random_action(state)
        return func(state)

    return randomized_action
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
    weighted score of current state with respect to is_max, and number of tricks this player has won.

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

class SimpleMCTSAgent(IAgent):
    """ Agent implementing simplified version of MCTS -
        only looks at end-results of simulation, without backpropogation.
        Our agent's local decision rule is decided by `action_chooser_function`, while
        the opponent's local decisions are chosen randomly."""

    def __init__(self, action_chooser_function='random_action', num_simulations=100):
        """

        :param str action_chooser_function: See `super().__init__()` docstring
        :param int num_simulations: How many simulations for rollout
        """

        self.action_chooser_function = lookup(action_chooser_function,
                                              globals())
        self.num_simulations_total = 0
        self.action_value = defaultdict(lambda: 0)  # type: Dict[Card, int]  # Maps values of playable actions
        self.num_simulations = num_simulations
        self.executor = ThreadPoolExecutor()
        super().__init__(None)

    def get_action(self, state):
        action = self.rollout(state, self.num_simulations)
        return action

    def rollout(self, state, num_simulations):
        """
        Performs `num_simulations` rollouts - i.e. stochastically simulate `num_simlations` games.

        :param State state: Current state of the game
        :param int num_simulations: How many games to simulate. Our agent's choices are made according to `action_chooser_function`
            while the opoonent's are chosen randomly.
        :returns Card: Best action
        """

        legal_actions = state.get_legal_actions()
        rollout_actions = np.random.choice(legal_actions,  # Pre-select initial actions
                                           size=num_simulations, replace=True)
        best_action = np.random.choice(legal_actions)

        # Simulate games on separate threads
        games = [SimulatedGame(SimpleAgent(self.action_chooser_function),
                               SimpleAgent('random_action'), False,
                               state, action) for action in rollout_actions]
        futures = [self.executor.submit(game.run) for game in games]
        futures_queue = Queue(num_simulations)
        for future in futures:
            futures_queue.put(future)

        # Poll threads for termination. Each future's return value is a boolean.
        while not futures_queue.empty():
            future = futures_queue.get()
            futures_queue.task_done()
            if future.running():
                futures_queue.put(future)
            else:
                assert future.result()

        # Collect results
        for game in games:
            assert game.winning_team != -1
            winning_team = game.teams[game.winning_team]
            if winning_team.has_player(state.curr_player):
                self.action_value[game.starting_action] += 1

            self.num_simulations_total += 1

        # Choose best action
        for action in legal_actions:
            best_action = action if self.action_value[action] > self.action_value[best_action] \
                else best_action

        return best_action


class StochasticSimpleMCTSAgent(SimpleMCTSAgent):
    """ Same as `SimpleMCTSAgent`, but with randomness injected into
        our agent's choices within simulations."""

    def __init__(self, action_chooser_function='random_action', num_simulations=100, epsilon=0.1):
        """

        :param action_chooser_function: See `super().__init__()` docstring
        :param num_simulations: See `super().__init__()` docstring
        :param float epsilon: Value in range [0,1]. w.p. `epsilon` our agent chooses random action.
        """

        assert 0 <= epsilon <= 1
        super().__init__(action_chooser_function, num_simulations)
        self.epsilon = epsilon
        self.action_chooser_function = add_randomness_to_action(self.action_chooser_function, self.epsilon)


class PureMCTSAgent(SimpleMCTSAgent):
    """ Implements the full MCTS algorithm, in context of Bridge."""

    def __init__(self, action_chooser_function='random_action', num_simulations=100):
        """

        :param str action_chooser_function: See `super().__init__()` docstring
        :param int num_simulations: How many simulations for rollout
        """

        self.action_chooser_function = lookup(action_chooser_function,
                                              globals())
        super().__init__(action_chooser_function, num_simulations)
        # self.root = None  # type: MCTSNode
        self.roots = {}  # type: Dict[Player, MCTSNode]  # One root for each of the agent's team players.

    def get_action(self, state):
        if not state.prev_tricks:
            # New game, first play for current player, create new root
            self.roots[state.curr_player] = MCTSNode(state)

        else:
            # Need to remove impossible paths from tree
            self.prune_tree(state)

        # Prepare tree for evaluation of best move
        root = self.roots[state.curr_player]
        for _ in range(0, self.num_simulations):
            # Exploration stage
            expanded_node = self.explore(root)
            # Rollout stage
            reward = expanded_node.rollout()
            # Backpropogation stage
            expanded_node.backpropagate(reward)

        # Exploitation stage
        best_child =  root.best_child(uct_param=1.4)
        best_action = best_child.parent_action

        return best_action

    def prune_tree(self, state):
        """
        Removes unreachable paths from tree, and updates root node.
        :param State state: current game state
        """

        if not state.prev_tricks:  # new game, no need for pruning
            return

        # Evaluates plays since last time current player played
        prev_trick = state.prev_tricks[-1]
        curr_trick = state.trick
        actions = []
        for player in state.players:
            action = prev_trick.get_card(player)
            if not action:
                action = curr_trick.get_card(player)
            assert action is not None
            actions.append(action)

        current_root = self.roots[state.curr_player]
        next_root = None
        # Traverse tree according to order of play.
        # Tree may not contain this path due to nature of MCTS, in which case we create a new tree.
        for turn in range(3):
            for child in current_root.children:
                if child.parent_action in actions:
                    next_root = child
                    break
            if next_root is None:
                break
            current_root = next_root
            actions.remove(current_root.parent_action)
            next_root = None

        if len(actions) != 1:
            # This means last three plays traversed down an unexpanded path
            self.roots[state.curr_player] = MCTSNode(state)
            return

        action = actions[0]
        for child in current_root.children:
            if child.player_pos == state.curr_player.position \
                and child.parent_action == action:
                next_root = child
                break

        if next_root is None:
            # This means the last move did not lead to current player,
            # so we need to create a new tree
            self.roots[state.curr_player] = MCTSNode(state)
            return

        # This path exists in old tree, make this node the root,
        # and eliminate illegal paths
        current_root = next_root
        assert current_root.player_pos == state.curr_player.position
        self._make_root_node(current_root, state.curr_player, state)

    def _make_root_node(self, node, player, state):
        """
        Updates `player`'s root node to be `node`.
        :param MCTSNode node: New root for `player`
        :param Player player: Player whose root we are changing
        :param State state: Current game state
        """

        node.parent_action = None
        node.parent = None
        new_children = []
        for child in node.children:
            if child.parent_action not in state.already_played:
                new_children.append(child)
        node.children = new_children

        untried_actions = set()
        tried_actions = set()
        for action in node.untried_actions:
            if action not in state.already_played:
                untried_actions.add(action)
            else:
                tried_actions.add(action)

        node._untried_actions = untried_actions
        node._tried_actions.update(tried_actions)
        self.roots[player] = node

    def explore(self, root):
        """
        Explores tree, choosing a leaf node for rollout stage. Expands leaf node if not terminal.
        :param MCTSNode root: Root to explore
        :returns MCTSNode: node on which to perform rollout
        """

        current_node = root
        while not current_node.is_terminal:
            if not current_node.is_fully_expanded:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

class MCTSNode:
    """ Node in search tree for Pure MCTS"""

    def __init__(self, state, parent=None,
                 parent_action=None, action_chooser_func='random_action'):
        """

        :param State state: current game state
        :param MCTSNode parent: Parent of node. If None, this node is assumed to be a root node.
        :param Card parent_action: If `parent` is not None,
            this is the action made by parent that lead to this node. Else, value is ignored.
        :param str action_chooser_function: See `super().__init__()` docstring
        """

        if isinstance(action_chooser_func, str):
            self.action_chooser_func = lookup(action_chooser_func, globals())
        else:
            self.action_chooser_func = action_chooser_func

        self.state = copy(state)
        self.parent = parent
        self.children = []
        if not self.parent:  # Is root node
            self.parent_action = None
            if self.state.teams[0].has_player(self.state.curr_player):
                self.team = self.state.teams[0]
            else:
                self.team = self.state.teams[1]

        else:
            self.team = parent.team
            self.parent_action = parent_action

        self._number_of_visits = 0.
        self._results = defaultdict(lambda: 0)  # type: Dict[int, int]  # maps team# to no. of wins
        self._untried_actions = None  # type: Set[Card]
        self._tried_actions = set()  # type: Set[Card]
        self.max_player = 1 if self.team.has_player(self.state.curr_player) else -1
        self.player_pos = state.curr_player.position


    def best_child(self, uct_param=1.4):
        """
        Returns child node with best UCT upper bound value
        :param float uct_param: Scaling factor for UCT value calculation. Default 1.4 ~ sqrt(2)
        :returns MCTSNode: best child node
        """

        choices_weights = [self.UCT_value(child, uct_param) for child in self.children]
        child_idx = int(np.argmax(choices_weights))
        return self.children[child_idx]

    def UCT_value(self, node, uct_param):
        """
        Calculates UCT value for node
        :param MCTSNode node: node for calculation
        :param float uct_param: Scaling factor for UCT value calculation
        :returns float: UCT value
        """

        return (node.q_value / node.num_visits) + \
               uct_param * np.sqrt((2 * np.log(self.num_visits) / node.num_visits))

    def rollout_policy(self, possible_moves):
        """
        Chooses "arm" of root node on which to perform rollout
            ("arm" as in the "Multi armed bandit" problem).
        Action is chosen at random.
        :param List[Card] possible_moves: List of "arms"
        :returns Card: chosen "arm", i.e. action to take for rollout
        """

        return np.random.choice(possible_moves)

    @property
    def untried_actions(self) -> List[Card]:
        """ List of actions still unexplored"""

        if self._untried_actions is None:
            self._untried_actions = set(self.state.curr_player.hand.cards)
        return list(self._untried_actions.intersection(self.state.get_legal_actions()))

    @property
    def q_value(self) -> int:
        """ Difference between wins and losses count for current node"""

        wins = self._results[self.max_player]
        loses = self._results[-1 * self.max_player]
        return wins - loses

    @property
    def num_visits(self) -> float:
        """ Number of time node was visited, updated each rollout"""

        return self._number_of_visits

    def expand(self):
        """ Expands a child node that wasn't explored yet.
            Assumes not all children were explored.
        :returns MCTSNode: Child node for exploration
        """

        action = self.untried_actions.pop()
        next_state = self.state.get_successor(action)
        assert action not in self.state.already_played
        assert action in self.state.curr_player.hand.cards
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        self._untried_actions.remove(action)
        self._tried_actions.add(action)
        return child_node

    @property
    def is_terminal(self) -> bool:
        """ Is current node a terminal node"""
        return self.state.is_game_over

    def rollout(self):
        """ Performs single rollout on current node -
            i.e. simulates a single game with current state as initial state. """

        if self.is_terminal:
            reward = 1 if self.team.has_player(self.state.curr_player) else -1
            return reward

        current_rollout_state = self.state
        possible_moves = current_rollout_state.get_legal_actions()
        action = self.rollout_policy(possible_moves)
        game = SimulatedGame(SimpleAgent(self.action_chooser_func),
                             SimpleAgent('random_action'), False,
                             current_rollout_state, action)
        assert game.run()
        winning_team_idx = game.winning_team
        winning_team = current_rollout_state.teams[winning_team_idx]
        reward = 1 if self.team == winning_team else -1
        return reward

    @property
    def is_fully_expanded(self) -> bool:
        """ Whether all children of node were previously expanded"""
        return len(self.untried_actions) == 0

    def backpropagate(self, result) -> None:
        """
        Backpropogates result of single rollout up the tree.
        :param int result: 1 if max player won in rollout, -1 if min player won.
        """

        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent is not None:
            self.parent.backpropagate(result)

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
                action = Card(card_number, card_suit, state.trump)
                legal_moves = state.get_legal_actions()
                if action in legal_moves:
                    return action
                else:
                    print(f"{card_suit, card_number} "
                          f"is not in your hand, try again")

            except ValueError or IndexError or TypeError:
                print(f"{inp} is not a valid action, try again")
