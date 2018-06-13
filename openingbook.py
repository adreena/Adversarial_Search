
###############################################################################
#                    YOU DO NOT NEED TO MODIFY THIS FILE                      #
###############################################################################
import argparse
import logging
import math
import os
import random
import textwrap
from pprint import pprint
import pickle
from multiprocessing.pool import ThreadPool as Pool

from isolation import Isolation, DebugState, Agent, play
from sample_players import RandomPlayer, GreedyPlayer, MinimaxPlayer
from my_custom_player import CustomPlayer

logger = logging.getLogger(__name__)

NUM_PROCS = 1
NUM_ROUNDS = 15  # number times to replicate the match; increase for higher confidence estimate
TIME_LIMIT = 150  # number of milliseconds before timeout

TEST_AGENTS = {
    "RANDOM": Agent(RandomPlayer, "Random Agent"),
    "GREEDY": Agent(GreedyPlayer, "Greedy Agent"),
    "MINIMAX": Agent(MinimaxPlayer, "Minimax Agent"),
    "SELF": Agent(CustomPlayer, "Custom TestAgent")
}





def build_table(num_rounds=NUM_ROUNDS):
    # Builds a table that maps from game state -> action
    # by choosing the action that accumulates the most
    # wins for the active player. (Note that this uses
    # raw win counts, which are a poor statistic to
    # estimate the value of an action; better statistics
    # exist.)
    from collections import defaultdict, Counter
    book = defaultdict(Counter)
    _board =  defaultdict(Counter)
    for _ in range(num_rounds):
        print(_)
        state = Isolation() #GameState()
        build_tree_random(state, book, _board)
        state = Isolation()
        build_tree_minmax(state, book, _board)
    result = {k: max(v, key=v.get) for k, v in book.items()}
    with open("data.pickle", 'wb') as f:
        pickle.dump(result, f)
    return result

def build_tree_random(state, book, _board, depth=4):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    action = random.choice(state.actions())
    reward = build_tree_random(state.result(action), book,_board, depth - 1)
    state_str = "{:b}".format(state.board)
    hash_key = (state_str, state.locs, state.player())
    book[hash_key][action] += reward
    _board[state_str] = state 
    return -reward

def build_tree_minmax(state, book, _board, depth=4):
    if depth <= 0 or state.terminal_test():
        return -simulate(state)
    action = minimax_search(state, depth)
    reward = build_tree_minmax(state.result(action), book,_board, depth - 1)
    state_str = "{:b}".format(state.board)
    hash_key = (state_str, state.locs, state.player())
    book[hash_key][action] += reward
    _board[state_str] = state 
    return -reward

def minimax_search(state, depth):

    def min_value(state, depth):
        # check for terminating state
        if state.terminal_test(): return state.utility(state.player())
        if depth <= 0 : return heuristic_score(state)
        value = float("inf")
        for action in state.actions():
            value = min(value, max_value(state.result(action), depth-1))
        return value
    
    def max_value(state, depth):
        # check for terminating state
        if state.terminal_test(): return state.utility(state.player())
        if depth <= 0 : return heuristic_score(state)

        value = float("-inf")
        for action in state.actions():
            value = max(value, min_value(state.result(action), depth-1))
        return value

    def heuristic_score(state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[state.player()]
        opp_loc = state.locs[1 - state.player()]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
    return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))



def simulate(state):
    player_id = state.player()
    while not state.terminal_test():
        state = state.result(random.choice(state.actions()))
        
    return -1 if state.utility(player_id) < 0 else 1


def main(args):
    test_agent = TEST_AGENTS[args.opponent.upper()]
    custom_agent = Agent(CustomPlayer, "Custom Agent")
    table = build_table(num_rounds=int(args.rounds))
    # wins, num_games = play_matches(custom_agent, test_agent, args)

    # logger.info("Your agent won {:.1f}% of matches against {}".format(
    #    100. * wins / num_games, test_agent.name))
    # print("Your agent won {:.1f}% of matches against {}".format(
    #    100. * wins / num_games, test_agent.name))
    # print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run matches to test the performance of your agent against sample opponents.",
        epilog=textwrap.dedent("""\
            Example Usage:
            --------------
            - Run 40 games (10 rounds = 20 games x2 for fair matches = 40 games) against
              the greedy agent with 4 parallel processes: 

                $python run_match.py -f -r 10 -o GREEDY -p 4

            - Run 100 rounds (100 rounds = 200 games) against the minimax agent with 1 process:

                $python run_match.py -r 100
        """)
    )
    parser.add_argument(
        '-f', '--fair_matches', action="store_true",
        help="""\
            Run 'fair' matches to mitigate differences caused by opening position 
            (useful for analyzing heuristic performance).  Setting this flag doubles 
            the number of rounds your agent will play.  (See README for details.)
        """
    )
    parser.add_argument(
        '-r', '--rounds', type=int, default=NUM_ROUNDS,
        help="""\
            Choose the number of rounds to play. Each round consists of two matches 
            so that each player has a turn as first player and one as second player.  
            This helps mitigate performance differences caused by advantages for either 
            player getting first initiative.  (Hint: this value is very low by default 
            for rapid iteration, but it should be increased significantly--to 50-100 
            or more--in order to increase the confidence in your results.
        """
    )
    parser.add_argument(
        '-o', '--opponent', type=str, default='MINIMAX', choices=list(TEST_AGENTS.keys()),
        help="""\
            Choose an agent for testing. The random and greedy agents may be useful 
            for initial testing because they run more quickly than the minimax agent.
        """
    )
    parser.add_argument(
        '-p', '--processes', type=int, default=NUM_PROCS,
        help="""\
            Set the number of parallel processes to use for running matches.  WARNING: 
            Windows users may see inconsistent performance using >1 thread.  Check the 
            log file for time out errors and increase the time limit (add 50-100ms) if 
            your agent performs poorly.
        """
    )
    parser.add_argument(
        '-t', '--time_limit', type=int, default=TIME_LIMIT,
        help="Set the maximum allowed time (in milliseconds) for each call to agent.get_action()."
    )
    args = parser.parse_args()

    logging.basicConfig(filename="matches.log", filemode="w", level=logging.DEBUG)
    logging.info(
        "Search Configuration:\n" +
        "Opponent: {}\n".format(args.opponent) +
        "Rounds: {}\n".format(args.rounds) +
        "Fair Matches: {}\n".format(args.fair_matches) +
        "Time Limit: {}\n".format(args.time_limit) +
        "Processes: {}".format(args.processes)
    )

    main(args)
