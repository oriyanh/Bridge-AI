# Double Dummy Bridge Solver
In this project we try to create a sophisticated computer agent to play the Contact Bridge card game. Our goal is to develop an agent that is tough to play against,  with fast reaction time so it is able to play in real time against humans. We approached this as a search problem, and implemented search-tree heuristics based on Minimax and Monte Carlo Tree Search.<br>
Implemented as a final project for the "Introduction to Aritifical Intelligence" course of the Hebrew University of Jerusalem.<br>
# Running Instructions
* Create a `virtualenv` with `python3.7` - `virtualenv -p python3.7 venv` 
* `pip install -r requirements.txt` to install project dependencies.
* To run a match, run `python3.7 match.py --agent1 <agent arguments> --agent2 <agent arguments> --num_games <int> --verbose_mode <0/1>`  where each agent encoding is of the form described in match.py’s documentation.

We encourage you to try and run match.py with some of the following arguments preferably from a console outside of an IDE (during the game press the Enter key to perform the next action):

Simple vs Simple:
* `--agent1 Random --agent2 Random`
* `--agent1 HighestFirst --agent2 SoftLongGreedy`

AlphaBeta vs Simple:
* `--agent1 AlphaBeta-ShortGreedyEvaluation-5 --agent2 LowestFirst`
* `--agent1 AlphaBeta-LongGreedyEvaluation-10 --agent2 Random`
* `--agent1 AlphaBeta-HandEvaluation-5 --agent2 HighestFirst`
* `--agent1 AlphaBeta-CountOfTricksWon-10 --agent2 HighestFirst`

MCTS vs Simple:
* `--agent1 MCTS-simple-HardLongGreedy-50 --agent2 HardShortGreedy`
* `--agent1 MCTS-stochastic-Random-500 --agent2 SoftShortGreedy`
* `--agent1 MCTS-pure-LowestFirst-250 --agent2 Random`

If you wish to run games automatically without seeing each game state and without pressing Enter after each move, add the argument “--verbose_mode 0”.
For more arguments, refer to `match.py`.
