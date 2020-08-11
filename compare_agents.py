import time

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt

from match import *

GAMES_PER_MATCH = 100

simple_func_names = [
    'highest_first_action',
    'lowest_first_action',
    'random_action',
    'hard_short_greedy_action',
    'hard_long_greedy_action',
    'soft_short_greedy_action',
    'soft_long_greedy_action'
]

simple_agent_names = [
    'HighestFirst',
    'LowestFirst',
    'Random',
    'HardGreedy',
    'HardGreedyExtended',
    'SoftGreedy',
    'SoftGreedyExtended'
]

ab_evaluation_func_names = [
                            'count_tricks_won_evaluation_function',
                            'greedy_evaluation_function1',
                            'greedy_evaluation_function2',
                            'hand_evaluation_heuristic',
                            ]

ab_evaluation_agent_names = ['AlphaBeta(#tricks won)',
                             'AlphaBeta(#legal moves)',
                             'AlphaBeta(#winning moves)',
                             'AlphaBeta(hand evaluation)',
                             ]


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
        ax.imshow(results_matrix, cmap='magma', vmin=0, vmax=100)
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

        title = f"Simple vs Simple Agents, y-axis win %, {GAMES_PER_MATCH} games"
        name = f"SingleActionAgent_{NUM_GAMES}games.png"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        plt.plot()
        plt.savefig(f"graphs/simple_agent/{name}.png")

    print()
    start_time = time.time()
    results = np.empty((len(simple_func_names),
                        len(simple_func_names)))
    results[:] = np.nan

    run_all_simple_actions_matches(results)
    display_table_simple_agents(results)
    print(f"--- Graph generation took "
          f"{int(time.time() - start_time)} seconds ---")


def compare_simple_agents_vs_ab_agents(depth, hand_size):
    def run_all_simple_agents_vs_ab_matches(depth, hand_size, results_matrix):
        for i in range(len(ab_evaluation_func_names)):
            for j in range(len(simple_func_names)):

                ab_agent = AlphaBetaAgent(ab_evaluation_func_names[i], depth)
                simple_agent = SimpleAgent(simple_func_names[j])
                print(f"{ab_evaluation_agent_names[i]} vs. "
                      f"{simple_agent_names[j]}")

                curr_match = Match(ab_agent, simple_agent,
                                   num_games=GAMES_PER_MATCH,
                                   verbose_mode=False,
                                   cards_in_hand=hand_size)
                curr_match.run()
                print(f"Score: {curr_match.games_counter[0]:02} -"
                      f" {curr_match.games_counter[1]:02}\n")
                results_matrix[i, j] = \
                    100 * curr_match.games_counter[0] / GAMES_PER_MATCH
        return results_matrix

    def display_table_simple_agents_vs_ab(depth, hand_size, results_matrix):
        fig, ax = plt.subplots(dpi=600)
        ax.imshow(results_matrix, cmap='magma', vmin=0, vmax=100)
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

        title = f"Simple vs Simple Agents, y-axis win %\n " \
                f"games:{GAMES_PER_MATCH} , hand size:{hand_size}, depth: {depth}"
        name = f"SingleActionAgent_{NUM_GAMES}games_{hand_size}cards_{depth}depth.png"
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        plt.plot()
        plt.savefig(f"graphs/alpha_beta/{name}.png")


    print()
    start_time = time.time()

    results = np.empty((len(ab_evaluation_func_names),
                        len(simple_func_names)))
    results[:] = np.nan

    run_all_simple_agents_vs_ab_matches(depth, hand_size, results)
    display_table_simple_agents_vs_ab(depth, hand_size, results)
    print(f"--- Graph generation took "
          f"{int(time.time() - start_time)} seconds ---")


# compare_simple_agents()
compare_simple_agents_vs_ab_agents(depth=5, hand_size=4)
compare_simple_agents_vs_ab_agents(depth=10, hand_size=4)
compare_simple_agents_vs_ab_agents(depth=15, hand_size=4)
compare_simple_agents_vs_ab_agents(depth=5, hand_size=8)
compare_simple_agents_vs_ab_agents(depth=10, hand_size=8)
compare_simple_agents_vs_ab_agents(depth=15, hand_size=8)
compare_simple_agents_vs_ab_agents(depth=4, hand_size=13)
