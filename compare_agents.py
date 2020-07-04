import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from match import *

GAMES_PER_MATCH = 100

all_agents = [HighestFirstAgent,
              LowestFirstAgent,
              RandomAgent,
              HardGreedyAgent,
              SoftGreedyAgent, ]

results = np.empty((len(all_agents), len(all_agents)))
results[:] = np.nan


def run_all_matches():
    # For n agents, will run n*(n-1)/2 matches

    for i in range(len(all_agents)):
        for j in range(i + 1, len(all_agents)):
            # For each pair of agents
            agent_0, agent_1 = all_agents[i], all_agents[j]
            print(f"{agent_0.__name__} vs. {agent_1.__name__}")

            # Run match
            curr_match = Match(agent_0(), agent_1(), GAMES_PER_MATCH, False)
            curr_match.run()

            # Print match result and update scores table
            print(f"Score: {curr_match.games_counter[0]:02} -"
                  f" {curr_match.games_counter[1]:02}\n")
            results[i, j] = 100 * curr_match.games_counter[0] / GAMES_PER_MATCH
            results[j, i] = 100 - results[i, j]


def display_table():
    all_agents_names = [c.__name__ for c in all_agents]
    fig, ax = plt.subplots(dpi=300)
    im = ax.imshow(results, cmap='plasma', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(all_agents_names)))
    ax.set_yticks(np.arange(len(all_agents_names)))
    ax.set_xticklabels(all_agents_names)
    ax.set_yticklabels(all_agents_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(all_agents_names)):
        for j in range(len(all_agents_names)):
            text = ax.text(
                j, i, f"{results[i, j]:05.2f}" if i != j else "",
                ha="center", va="center", color="w")
            text.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground='black'),
                 path_effects.Normal()])
    ax.set_title("Win rate % (of agent on y-axis)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    plt.show()


print()
run_all_matches()
display_table()
