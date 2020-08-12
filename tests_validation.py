import os

from validation import *
from multi_agents import SimpleAgent, AlphaBetaAgent, SimpleMCTSAgent, \
    StochasticSimpleMCTSAgent, PureMCTSAgent
from compare_agents import simple_func_names, simple_agent_names, \
    ab_evaluation_func_names, ab_evaluation_agent_names
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
                    num_of_games: int, min_tricks_num: int=0):
    # labels = [i for i in range(min_tricks_num + 1, 13)]
    labels = [i for i in range(13 - min_tricks_num, 1, -1)]
    x = np.arange(len(labels))  # the label locations
    num_agents = len(agents_scores)
    width = 1 / (num_agents+2)  # the width of the bars

    fig, ax = plt.subplots()
    rects_list = []
    for i in range(num_agents):
        rects_list.append(ax.bar(x + ((i - ((num_agents-1) / 2)) * width),
                                 agents_scores[i], width, label=names[i]))

    ax.set_ylabel('Match success rate (%)')
    ax.set_xlabel('Hand size (cards)')
    ax.set_title(f'Scores by trick and agent\n (data from {num_of_games} games)')
    ax.set_title(f'Prediction success rate as func. of # cards in hand\naveraged over {num_of_games} games')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='best')
    ax.set_ylim([0, 119])
    fig.tight_layout()
    plt.show()


def run_validation(agents_list: List[IAgent], num_of_games: int,
                   min_tricks: int=0) -> List[np.ndarray]:
    """

    :param agents_list:
    :param num_of_games: if 0: run all games
    :param min_tricks: trick index to start validation from (0 for start of game)
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
            curr_checks, curr_succeeds = validate_agent_per_data_game(
                agent, games.games[i], min_tricks)
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

    num_of_games = 1  # how many data_games to use for validation
    minimum_round = 8  # trick index to start validation from (0 for start of game)

    # Run Simple agents
    # agents_list = [SimpleAgent(kind) for kind in simple_func_names]
    # agents_scores_list = run_validation(agents_list, num_of_games, min_tricks_num)
    # display_results(agents_scores_list, simple_agent_names, num_of_games)

    # Run AB agents
    # agents_list = [AlphaBetaAgent(kind, depth=12) for kind in ab_evaluation_func_names]
    # agents_scores_list = run_validation(agents_list, num_of_games, min_tricks_num)
    # display_results([agent[min_tricks_num:] for agent in agents_scores_list],
    #                 ab_evaluation_agent_names, num_of_games, min_tricks_num)

    # Run MTCs (choose one agents_list every time not to be a comment)
    num_of_simulations = 10
    agents_list = [SimpleMCTSAgent(kind, num_simulations=num_of_simulations)
                   for kind in simple_func_names]
    agent_names = [f"{agent_cls.__class__.__name__}({func_name})" for agent_cls, func_name in zip(agents_list, simple_func_names)]
    # agents_list = [StochasticSimpleMCTSAgent(kind, num_simulations=num_of_simulations)
    #                for kind in simple_func_names]
    # agents_list = [PureMCTSAgent(kind, num_simulations=num_of_simulations)
    #                for kind in simple_func_names]
    agents_scores_list = run_validation(agents_list, num_of_games, minimum_round)
    display_results([agent[minimum_round:] for agent in agents_scores_list],
                    agent_names, num_of_games, minimum_round)
