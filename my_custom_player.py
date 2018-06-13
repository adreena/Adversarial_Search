
from sample_players import DataPlayer
import math
import random
from pprint import pprint
import pickle

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """

    def get_action(self, state, match_id):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        # import random
        # print(state.ply_count)

        # it's the begining state
        # random_action = random.choice(state.actions())
        
        
        # detect opening move
        open_move = None
        # if None in state.locs:
        #     state_str = "{0:b}".format(state.board)
        #     hash_key = (state_str, state.locs[0], state.locs[1], self.player_id)
        #     if hash_key in self.data.keys():
        #         open_move = self.data[hash_key]
                
        if open_move is not None:
            # print('\nopen move applied:', open_move)
            self.queue.put(open_move)
        else:
            if state.ply_count < 2: self.queue.put(random.choice(state.actions()))

            self.queue.put(self.minimax_search(state, depth=4))

        # self.queue.put(self.alpha_beta_search(state, depth=4))
    
    def minimax_search(self, state, depth):
        

        def min_value(state, depth):
            # check for terminating state
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.heuristic_score(state)
            value = float("inf")
            for action in state.actions():
                new_state = state.result(action)
                value = min(value, max_value(new_state, depth-1))
            return value
        

        def max_value(state, depth):
            # check for terminating state
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.heuristic_score(state)

            value = float("-inf")
            for action in state.actions():
                new_state = state.result(action)
                value = max(value, min_value(new_state, depth-1))
            return value

        best_action = max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1))

        return best_action

    def alpha_beta_search(self, state, depth):

        def min_value(state, alpha, beta, depth):
            # check for terminating state
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.heuristic_score(state)
            value = float("inf")
            open_book_action = get_data_openbook(state)
            if open_book_action != None:
                value = min(value, max_value(state.result(open_book_action), alpha, beta, depth-1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            else:
                for action in state.actions():
                    value = min(value, max_value(state.result(action), alpha, beta, depth-1))
                    if value <= alpha:
                        return value
                    beta = min(beta, value)
            return value
        

        def max_value(state, alpha, beta, depth):
            # check for terminating state
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0 : return self.heuristic_score(state)

            value = float("-inf")
            open_book_action =  get_data_openbook(state)
            if open_book_action != None:
                value = max(value, min_value(state.result(open_book_action), alpha, beta, depth-1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            else:
                for action in state.actions():
                    value = max(value, min_value(state.result(action), alpha, beta, depth-1))
                    if value >= beta:
                        return value
                    alpha = max(alpha, value)
            return value

        def get_data_openbook(state):
            state_str = "{:b}".format(state.board)
            hash_key = (state_str, state.locs, state.player())
            if hash_key in self.data.keys():
                return self.data[hash_key]
            else :
                return None
        
        return max(state.actions(), key=lambda x: min_value(state.result(x), float('-inf'), float('inf'), depth - 1))

    def heuristic_score(self, state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)


    