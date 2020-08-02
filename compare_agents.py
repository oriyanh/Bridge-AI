import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from match import *

GAMES_PER_MATCH = 100

all_single_action_names = ['HighestFirstAgent',
              'LowestFirstAgent',
              'RandomAgent',
              'HardGreedyAgent',
              'SoftGreedyAgent', ]

all_single_action_func = ['highest_first_action',
              'lowest_first_action',
              'random_action',
              'hard_greedy_action',
              'soft_greedy_action', ]

all_ab_evaluation_func = ['is_target_reached_evaluation_function',
                          'count_tricks_won_evaluation_function']

all_ab_evaluation_names = ['reach target',
                          'count of tricks won']

results = np.empty((len(all_single_action_func), len(all_single_action_func)))
results[:] = np.nan


def run_all_single_action_matches():
    for i in range(len(all_single_action_func)):
        for j in range(len(all_single_action_func)):
            # For each pair of agents
            agent0, agent1 = all_single_action_func[i], all_single_action_func[j]
            print(f"{all_single_action_names[i]} vs. {all_single_action_names[j]}")

            # Run match
            curr_match = Match(SingleActionAgent(agent0), SingleActionAgent(agent1),
                               GAMES_PER_MATCH, False)
            curr_match.run()

            # Print match result and update scores table
            print(f"Score: {curr_match.games_counter[0]:02} -"
                  f" {curr_match.games_counter[1]:02}\n")
            results[i, j] = 100 * curr_match.games_counter[0] / GAMES_PER_MATCH


def display_table_single_action():
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(results, cmap='plasma', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(all_single_action_names)))
    ax.set_yticks(np.arange(len(all_single_action_names)))
    ax.set_xticklabels(all_single_action_names)
    ax.set_yticklabels(all_single_action_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(all_single_action_names)):
        for j in range(len(all_single_action_names)):
            text = ax.text(
                j, i, f"{results[i, j]:05.2f}",
                ha="center", va="center", color="w")
            text.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground='black'),
                 path_effects.Normal()])
    ax.set_title("Win rate % (of agent on y-axis)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.show()


def compare_single_action_agents():
    print()
    run_all_single_action_matches()
    display_table_single_action()


def compare_ab_vs_ab_agents():
    agent0, agent1 = all_ab_evaluation_func[1], all_ab_evaluation_func[1]
    depth0, depth1 = 2, 2
    print(f"{all_ab_evaluation_names[1]} vs. {all_ab_evaluation_names[1]}")

    # Run match
    curr_match = Match(AlphaBetaAgent(evaluation_function=agent0,
                                      depth=depth0),
                       AlphaBetaAgent(evaluation_function=agent1,
                                      depth=depth1), GAMES_PER_MATCH, False)
    curr_match.run()

    # Print match result and update scores table
    print(f"Score: {curr_match.games_counter[0]:02} - {curr_match.games_counter[1]:02}\n")


def run_all_single_action_vs_ab_matches(depth):
    for i in range(len(all_single_action_func)):
        agent0 = all_single_action_func[i]
        print(f"{all_single_action_names[i]} vs. alfa beta")
        curr_match = Match(SingleActionAgent(agent0), AlphaBetaAgent(depth=depth),
                           GAMES_PER_MATCH, False)
        curr_match.run()
        print(f"Score: {curr_match.games_counter[0]:02} -"
              f" {curr_match.games_counter[1]:02}\n")
        results[i, 0] = 100 * curr_match.games_counter[0] / GAMES_PER_MATCH

        print(f"alfa beta vs. {all_single_action_names[i]}")
        curr_match = Match(AlphaBetaAgent(depth=depth), SingleActionAgent(agent0),
                           GAMES_PER_MATCH, False)
        curr_match.run()
        print(f"Score: {curr_match.games_counter[0]:02} -"
              f" {curr_match.games_counter[1]:02}\n")
        results[i, 1] = 100 * curr_match.games_counter[0] / GAMES_PER_MATCH

        # Print match result and update scores table


def display_table_single_action_vs_ab(depth):
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(results, cmap='plasma', vmin=0, vmax=100)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(len(all_single_action_names)))
    ax.set_xticklabels(['single action first', 'alfa beta first'])
    ax.set_yticklabels(all_single_action_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(all_single_action_names)):
        for j in range(2):
            text = ax.text(
                j, i, f"{results[i, j]:05.2f}",
                ha="center", va="center", color="w")
            text.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground='black'),
                 path_effects.Normal()])
    ax.set_title(f"Win rate % for y axis vs alfa-beta, depth: {depth}",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.show()


def compare_single_action_vs_ab_agents():
    print()
    depth = 4
    run_all_single_action_vs_ab_matches(depth)
    display_table_single_action_vs_ab(depth)


compare_single_action_vs_ab_agents()