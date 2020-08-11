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
    'HardShortGreedy',
    'HardLongGreedy',
    'SoftShortGreedy',
    'SoftLongGreedy'
]

ab_evaluation_func_names = ['greedy_evaluation_function1',
                            'greedy_evaluation_function2',
                            'hand_evaluation_heuristic',
                            'count_tricks_won_evaluation_function',
                            ]

ab_evaluation_agent_names = ['ShortGreedyEvaluation',
                             'LongGreedyEvaluation',
                             'HandEvaluation',
                             'CountOfTricksWon']


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
                                       cards_in_hand=5)
                else:
                    curr_match = Match(simple_agent, ab_agent,
                                       num_games=GAMES_PER_MATCH,
                                       verbose_mode=False,
                                       cards_in_hand=5)

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


# compare_simple_agents()
compare_simple_agents_vs_ab_agents(depth=5, ab_first=True)
compare_simple_agents_vs_ab_agents(depth=5, ab_first=False)
compare_simple_agents_vs_ab_agents(depth=10, ab_first=True)
compare_simple_agents_vs_ab_agents(depth=10, ab_first=False)
compare_simple_agents_vs_ab_agents(depth=15, ab_first=True)
compare_simple_agents_vs_ab_agents(depth=15, ab_first=False)


