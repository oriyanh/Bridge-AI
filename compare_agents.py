import time

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from match import *
from multi_agents import simple_func_names, simple_agent_names, \
    ab_evaluation_func_names, ab_evaluation_agent_names

NUM_CARDS = 13
GAMES_PER_MATCH = 100



def compare_simple_agents():
    def run_all_simple_actions_matches(results_matrix):
        for i in range(len(simple_func_names)):
            for j in range(len(simple_func_names)):
                # For each pair of agents
                agent0, agent1 = \
                    simple_func_names[i], \
                    simple_func_names[j]
                print(f"{simple_agent_names[i]} vs. "
                      f"{simple_agent_names[j]}")

                # Run match
                curr_match = Match(SimpleAgent(agent0),
                                   SimpleAgent(agent1),
                                   GAMES_PER_MATCH, False)
                curr_match.run()

                # Print match result and update scores table
                print(f"Score: {curr_match.games_counter[0]:02} -"
                      f" {curr_match.games_counter[1]:02}\n")
                results_matrix[i, j] = 100 * curr_match.games_counter[
                    0] / GAMES_PER_MATCH
        return results_matrix

    def display_table_simple_agents(results_matrix):
        fig, ax = plt.subplots(dpi=600)
        ax.imshow(results_matrix, cmap='plasma', vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(simple_agent_names)))
        ax.set_yticks(np.arange(len(simple_agent_names)))
        ax.set_xticklabels(simple_agent_names)
        ax.set_yticklabels(simple_agent_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(len(simple_agent_names)):
            for j in range(len(simple_agent_names)):
                text = ax.text(
                    j, i, f"{results_matrix[i, j]:04.1f}",
                    ha="center", va="center", color="w")
                text.set_path_effects(
                    [path_effects.Stroke(linewidth=2, foreground='black'),
                     path_effects.Normal()])

        title = "Simple vs Simple Agents\n" + \
                f"y-axis win %, higher is better"
        ax.set_title(title, fontsize=16, fontweight="bold")
        fig.tight_layout()
        plt.show()

    print()
    start_time = time.time()
    results = np.empty((len(simple_func_names),
                        len(simple_func_names)))
    results[:] = np.nan

    run_all_simple_actions_matches(results)
    display_table_simple_agents(results)
    print(f"--- Graph generation took "
          f"{int(time.time() - start_time)} seconds ---")


def compare_simple_agents_vs_ab_agents(depth, ab_first=True):
    def run_all_simple_agents_vs_ab_matches(depth, results_matrix):
        for i in range(len(ab_evaluation_func_names)):
            for j in range(len(simple_func_names)):

                ab_agent = AlphaBetaAgent(ab_evaluation_func_names[i], depth)
                simple_agent = SimpleAgent(simple_func_names[j])
                print(f"{ab_evaluation_agent_names[i]} vs. "
                      f"{simple_agent_names[j]}")

                if ab_first:
                    curr_match = Match(ab_agent, simple_agent,
                                       num_games=GAMES_PER_MATCH,
                                       verbose_mode=False,
                                       cards_in_hand=NUM_CARDS)
                else:
                    curr_match = Match(simple_agent, ab_agent,
                                       num_games=GAMES_PER_MATCH,
                                       verbose_mode=False,
                                       cards_in_hand=NUM_CARDS)

                curr_match.run()
                print(f"Score: {curr_match.games_counter[0]:02} -"
                      f" {curr_match.games_counter[1]:02}\n")
                results_matrix[i, j] = \
                    100 * curr_match.games_counter[0] / GAMES_PER_MATCH

        if not ab_first:
            results_matrix = 100 - results_matrix
        return results_matrix

    def display_table_simple_agents_vs_ab(depth, results_matrix):
        fig, ax = plt.subplots(dpi=600)
        ax.imshow(results_matrix, cmap='plasma', vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(simple_agent_names)))
        ax.set_yticks(np.arange(len(ab_evaluation_agent_names)))
        ax.set_xticklabels(simple_agent_names)
        ax.set_yticklabels(ab_evaluation_agent_names)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(len(simple_agent_names)):
            for j in range(len(ab_evaluation_agent_names)):
                text = ax.text(
                    i, j, f"{results_matrix[j, i]:04.1f}",
                    ha="center", va="center", color="w")
                text.set_path_effects(
                    [path_effects.Stroke(linewidth=2, foreground='black'),
                     path_effects.Normal()])

        title = f"AlphaBeta vs Simple Agents\n" \
                f"{'AlphaBeta' if ab_first else 'Simple'} agent play first\n" \
                f"y-axis win %, depth={depth}"

        ax.set_title(title, fontsize=16, fontweight="bold")
        fig.tight_layout()
        plt.show()

    print()
    start_time = time.time()

    results = np.empty((len(ab_evaluation_func_names),
                        len(simple_func_names)))
    results[:] = np.nan

    run_all_simple_agents_vs_ab_matches(depth, results)
    display_table_simple_agents_vs_ab(depth, results)
    print(f"--- Graph generation took "
          f"{int(time.time() - start_time)} seconds ---")

def run_all_simple_agents_vs_mcts(agent_cls, num_cards=13, **kwargs):
    results = np.empty((len(simple_func_names),
                        len(simple_func_names)))
    results[:] = np.nan
    for i in range(len(simple_func_names)):
        for j in range(len(simple_func_names)):
            agent0 = simple_func_names[i]
            print(f"{agent_cls.__name__} ({simple_func_names[j]} rule) vs. {simple_agent_names[i]}")
            curr_match = Match(agent_cls(simple_func_names[j], **kwargs),
                               SimpleAgent(agent0),
                               GAMES_PER_MATCH, False, cards_in_hand=num_cards)
            curr_match.run()
            print(f"Score: {curr_match.games_counter[0]:02} -"
                  f" {curr_match.games_counter[1]:02}\n")
            results[j, i] = 100 * curr_match.games_counter[0] / GAMES_PER_MATCH
    display_table_simple_agents_vs_MCTS(results, agent_cls, num_cards=num_cards, **kwargs)

def display_table_simple_agents_vs_MCTS(results, agent_cls, num_simulations=1, epsilon=None, num_cards=13):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(results, cmap='plasma', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(simple_agent_names)))
    ax.set_yticks(np.arange(len(simple_agent_names)))
    mcts_agent_names = [f"{agent_cls.__name__}({name})" for name in simple_agent_names]
    title = f"Win % of {agent_cls.__name__} vs normal agent\n" \
            f"{NUM_GAMES} games, {num_cards} cards, {num_simulations} sims."
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticklabels(simple_agent_names, fontsize=8)
    ax.set_yticklabels(mcts_agent_names, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(simple_agent_names)):
        for j in range(len(simple_agent_names)):
            text = ax.text(
                j, i, f"{results[i, j]:3.0f}%",
                ha="center", va="center", color="w", fontsize=8)
            text.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground='black'),
                 path_effects.Normal()])
    fig.tight_layout()
    plot_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    plot_fname = os.path.join(plot_dir, f"{agent_cls.__name__}_"
                                        f"_{NUM_GAMES}games_{num_cards}cards_{num_simulations}sims"
                                        f"{f'_{epsilon:1.2}eps' if epsilon else ''}.jpg")
    plt.tight_layout()
    plt.savefig(plot_fname)
    plt.show()

LEGAL_AGENT_TYPES =["AB", "MCTS", "SIMPLE"]
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--agent', required=True, type=str, choices=LEGAL_AGENT_TYPES, dest='agent')
    parser.add_argument('--games', default=100, type=int)
    parser.add_argument('--simulations', default=1000, type=int)
    parser.add_argument('--cards', default=5, type=int)
    parser.add_argument('--depth', default=5, type=int)
    parser.add_argument('--epsilon', default=0., type=float)
    args = parser.parse_args()
    return args

def mcts_main(num_games, num_simulations, num_cards, epsilon):
    global GAMES_PER_MATCH
    GAMES_PER_MATCH = num_games
    MCTS_classes = [SimpleMCTSAgent, PureMCTSAgent]
    for agent_cls in MCTS_classes:
        run_all_simple_agents_vs_mcts(agent_cls, num_simulations=num_simulations, num_cards=num_cards)

    run_all_simple_agents_vs_mcts(StochasticSimpleMCTSAgent, num_simulations=num_simulations, num_cards=num_cards, epsilon=epsilon)

def ab_main(num_games, num_cards, depth):
    global GAMES_PER_MATCH, NUM_CARDS
    GAMES_PER_MATCH = num_games
    NUM_CARDS = num_cards
    compare_simple_agents_vs_ab_agents(depth=5, ab_first=True)
    compare_simple_agents_vs_ab_agents(depth=5, ab_first=False)
    compare_simple_agents_vs_ab_agents(depth=10, ab_first=True)
    compare_simple_agents_vs_ab_agents(depth=10, ab_first=False)
    compare_simple_agents_vs_ab_agents(depth=15, ab_first=True)
    compare_simple_agents_vs_ab_agents(depth=15, ab_first=False)

if __name__ == '__main__':
    args = parse_args()
    if args.agent.upper() == 'MCTS':
        mcts_main(args.games, args.simulations, args.cards, args.epsilon)
    if args.agent.upper() == 'AB':
        ab_main(args.games, args.cards, args.depth)
    if args.agent.upper() == 'SIMPLE':
        compare_simple_agents()
