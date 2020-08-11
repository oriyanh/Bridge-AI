import os

from validation import *
from multi_agents import SimpleAgent
from compare_agents import simple_func_names, simple_agent_names
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def test_snapshot():
    file = "bridge_data_1/Aulpll23.pbn"
    # file = "bridge_data_1/Dkofro20.pbn"
    files = Parser([file])
    # out = files.games[0].snapshot(0, POSITIONS[0])
    # out = files.games[0].snapshot(0, POSITIONS[2])
    # out = files.games[0].snapshot(1, POSITIONS[2])
    out = files.games[0].snapshot(1, POSITIONS[0])
    # out = games[0].snapshot(6, POSITIONS[0])
    return out


def test_all_snapshots_single_game():
    file = "bridge_data_1/Aulpll23.pbn"
    games = Parser([file])
    # out = games.games[0].all_relevant_snapshots()
    # out = games.games[2].all_relevant_snapshots()
    # out = games.games[4].all_relevant_snapshots()
    out = games.games[6].all_relevant_snapshots()
    return out


def load_all_game():
    games_dir = os.path.join(os.getcwd(), 'bridge_data_1')
    files_name = os.listdir(games_dir)
    # files_name = []
    # for (dirpath, dirnames, filenames) in walk(games_dir):
    #     files_name.extend(filenames)
    #     break
    pbn_files = []
    for file in files_name:
        if file[-3:] == 'pbn':
            pbn_files.append(os.path.join('bridge_data_1', file))
    files = Parser(pbn_files)
    print("succeed loading")
    return files


def test_valid_games():
    games = load_all_game()
    for i, g in enumerate(games.games):
        trick_idx = max(0, len(g.tricks) - 1)
        g.snapshot(trick_idx, POSITIONS[0])
    print("all are valid")


def test_all_snapshots():
    games = load_all_game()
    for i, g in enumerate(games.games):
        g.all_relevant_snapshots()
    print("all snapshots for all games done")
    return


def games_length():
    games = load_all_game()
    histogram = np.zeros(12)
    for g in games.games:
        histogram[len(g.tricks) - 1] += 1
    print(histogram)
    plt.bar(np.arange(12) + 1, histogram)
    plt.title("Number of Tricks Histogram")
    plt.xlabel("# tricks")
    plt.ylabel("# games")
    plt.show()


def test_validate_agent_action():
    curr_agent = SimpleAgent('highest_first_action')

    file = "bridge_data_1/Aulpll23.pbn"
    files = Parser([file])
    return validate_agent_action(dg=files.games[4],
                                 trick_idx=1,
                                 position=PositionEnum.N,
                                 agent=curr_agent)


def test_validate_agent_by_the_whole_game():
    curr_agent = SimpleAgent('highest_first_action')

    file = "bridge_data_1/Aulpll23.pbn"
    files = Parser([file])
    return validate_agent_per_data_game(agent=curr_agent, dg=files.games[4])


def display_results(agents_scores: List[np.ndarray], names: List[str],
                    num_of_games: int):
    labels = [i for i in range(1, 13)]
    x = np.arange(len(labels))  # the label locations
    length = len(agents_scores)
    width = 1 / (length+2)  # the width of the bars

    fig, ax = plt.subplots()
    rects_list = []
    for i in range(length):
        rects_list.append(ax.bar(x + ((i - ((length-1) / 2)) * width),
                                 agents_scores[i], width, label=names[i]))

    ax.set_ylabel('Percents (%)')
    ax.set_xlabel('# tricks')
    ax.set_title(f'Scores by trick and agent\n (data from {num_of_games} games)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()


def run_validation(agents_list: List[IAgent], num_of_games: int) -> \
        List[np.ndarray]:
    """

    :param agents_list:
    :param num_of_games: if 0: run all games
    :return: list of arrays, one for every agent. the array is consist of 12
        entries, each represents the score of agent respectivly to the number
        of remaining cards in its hand.
    """
    agents_scores = [np.zeros(12) for _ in agents_list]
    games = load_all_game()
    if num_of_games == 0:
        num_of_games = len(games.games)

    for agent_idx, agent in enumerate(agents_list):
        start = perf_counter()
        all_checks, all_succeeds = np.zeros(12), np.zeros(12)
        for i in range(num_of_games):
            curr_checks, curr_succeeds = validate_agent_per_data_game(agent, games.games[i])
            all_checks += curr_checks
            all_succeeds += curr_succeeds
        agents_scores[agent_idx] = (all_succeeds / all_checks) * 100
        print(f"agent {agent_idx}: {(perf_counter() - start)/num_of_games} seconds/game, {num_of_games} games")
    print(agents_scores)
    return agents_scores


if __name__ == '__main__':
    # test_snapshot()
    # test_all_snapshots_single_game()
    # test_validate_agent_action()
    # test_load_all_game()
    # test_valid_games()
    # test_all_snapshots()
    # games_length()
    # test_validate_agent_by_the_whole_game()

    simple_agents = [SimpleAgent(kind) for kind in simple_func_names]
    num_of_games = 1000
    agents_scores_list = run_validation(simple_agents, num_of_games)
    display_results(agents_scores_list, simple_agent_names, num_of_games)
