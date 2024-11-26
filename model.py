#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
November 2024
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import itertools

from agent import Agent
from display import display_position, display_board


class Environment:
    """Generic environment. The reward only depends on the reached state."""
    def __init__(self):
        self.state = self.init_state()

    @staticmethod
    def is_game():
        return False
        
    @staticmethod
    def init_state():
        return None

    def reinit_state(self, state=None):
        if state is None:
            self.state = self.init_state()  
        else:
            self.state = deepcopy(state)
    
    @staticmethod
    def get_all_states():
        """Get all states."""
        raise ValueError("Not available. The state space might be too large.")
    
    @staticmethod
    def get_all_actions():
        """Get all actions."""
        actions = []
        return []

    @staticmethod
    def get_actions(state):
        """Get actions in a given state."""
        actions = []
        return actions

    @staticmethod
    def get_transition(state, action):
        """Get transition from a given state and action (distribution of next state)."""
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        """Get reward in reached state."""
        return 0

    @staticmethod
    def is_terminal(state):
        """Test if some state is terminal."""
        return False

    @staticmethod
    def encode(state):
        """Encode a state (making it hashable)."""
        return tuple(state)
    
    @staticmethod
    def decode(state):
        """Decode a state."""
        return np.array(state)
    
    def get_model(self, state, action):
        """Get the model from a given state and action (transition probabilities, next states and rewards)."""
        probs, states = self.get_transition(state, action)
        rewards = [self.get_reward(state) for state in states]
        return probs, states, rewards
    
    def step(self, action):
        """Apply action, get reward and modify state. Check whether the new state is terminal."""
        reward = 0
        stop = True
        if action is not None or action in self.get_actions(self.state):
            probs, states, rewards = self.get_model(self.state, action)
            i = np.random.choice(len(probs), p=probs)
            state = states[i]
            self.state = state
            reward = rewards[i]
            stop = self.is_terminal(state)
        return reward, stop

    def display(self, states=None):
        """Display current states or animate the sequence of states if available."""
        return None
                
                
class Walk(Environment):
    """Walk in 2D space."""

    Size = (5, 5)
    Rewards = {(1, 1): 1, (1, 3): -1, (3, 1): -1, (3, 3): 3}  
    Wind = {(0, 1): 0.1, (1, 0): 0.2}  # move probabilities due to wind

    @classmethod
    def set_parameters(cls, size, rewards, wind):
        cls.Size = size
        cls.Rewards = rewards
        if sum(list(wind.values())) >= 1:
            raise ValueError("The sum of probabilities must be less than 1.")
        cls.Wind = wind

    @staticmethod
    def init_state():
        return np.array([0, 0])
    
    @staticmethod
    def is_valid(state):
        n, m = Walk.Size
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m
 
    @staticmethod
    def get_all_states():
        n, m = Walk.Size
        states = [np.array([x,y]) for x in range(n) for y in range(m)]
        return states
        
    @staticmethod
    def get_all_actions():
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return actions
        
    @staticmethod
    def get_actions(state):
        all_actions = Walk.get_all_actions()
        actions = [action for action in all_actions if Walk.is_valid(state + action)]
        return actions

    @staticmethod
    def get_transition(state, action):
        next_state = state + action
        probs = []
        states = []
        for action, prob in Walk.Wind.items():
            new_state = next_state + action
            if Walk.is_valid(new_state):
                probs.append(prob)
                states.append(new_state)
        probs.append(1 - sum(probs))
        states.append(next_state)
        return probs, states
 
    @staticmethod
    def get_reward(state):
        reward = 0
        if tuple(state) in Walk.Rewards:
            reward = Walk.Rewards[tuple(state)]
        return reward
    
    def display(self, states=None, marker='o', marker_size=300, color_dict={'+': 'g', '-': 'r', '0': 'b'}, interval=200):
        
        def get_color(reward):
            if reward > 0:
                return color_dict['+']
            elif reward < 0:
                return color_dict['-']
            else:
                return color_dict['0']
        
        shape = (*self.Size, 3)
        image = 200 * np.ones(shape).astype(int)
        if states is None:
            marker_color = get_color(self.get_reward(self.state))
        else:
            marker_color = [get_color(self.get_reward(state)) for state in states]
           
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)

    @staticmethod
    def display_values(values):
        image = np.zeros(Walk.Size)
        values_scaled = np.array(values)
        values_scaled -= np.min(values)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        states = Walk.get_all_states()
        for state, value in zip(states, values_scaled):
            image[tuple(state)] = value 
            plt.imshow(image, cmap='gray');
            plt.axis('off')

    @staticmethod
    def display_policy(policy):
        image = np.zeros(Walk.Size)
        plt.imshow(image, cmap='gray')
        states = Walk.get_all_states()
        for state in states:
            _, actions = policy(state)
            action = actions[0]
            if action == (0, 0):
                plt.scatter(state[1], state[0], s=100, c='b')
            else:
                plt.arrow(state[1], state[0] , action[1], action[0], color='r', width=0.15, length_includes_head=True)
        plt.axis('off')


class Maze(Environment):
    """Maze."""

    Map = np.ones((2, 2)).astype(int)
    Start_State = (0, 0)
    Exit_States = [(1, 1)]

    @classmethod
    def set_parameters(cls, maze_map, start_state, exit_states):
        cls.Map = maze_map
        cls.Start_State = start_state
        cls.Exit_States = exit_states

    @staticmethod
    def init_state():
        return np.array(Maze.Start_State)

    @staticmethod
    def is_valid(state):
        n, m = Maze.Map.shape
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m and Maze.Map[x, y]
    
    @staticmethod
    def get_all_states():
        n, m = Maze.Map.shape
        states = [np.array([x, y]) for x in range(n) for y in range(m) if Maze.is_valid(np.array([x, y]))]
        return states
    
    @staticmethod
    def get_all_actions():
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return actions
    
    @staticmethod
    def get_actions(state):
        all_actions = Maze.get_all_actions()
        actions = [action for action in all_actions if Maze.is_valid(state + action)]
        return actions

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [state.copy() + action]
        return probs, states

    @staticmethod
    def get_reward(state):
        return -1

    @staticmethod
    def is_terminal(state):
        return tuple(state) in Maze.Exit_States

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        shape = (*Maze.Map.shape, 3)
        image = np.zeros(shape).astype(int)
        for i in range(3):
            image[:, :, i] = 255 * Maze.Map
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)

    @staticmethod
    def display_values(values):
        image = np.zeros(Maze.Map.shape)
        values_scaled = np.array(values)
        if np.min(values_scaled):
            values_scaled /= np.min(values_scaled)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        states = Maze.get_all_states()
        for state, value in zip(states, values_scaled):
            image[tuple(state)] = 1 - 0.8 * value 
            plt.imshow(image, cmap='gray');
            plt.axis('off')

    @staticmethod
    def display_policy(policy):
        image = np.zeros(Maze.Map.shape)
        states = Maze.get_all_states()
        for state in states:
            image[tuple(state)] = 1
        plt.imshow(image, cmap='gray')
        for state in states:
            if not Maze.is_terminal(state):
                _, actions = policy(state)
                action = actions[0]
                if action == (0,0):
                    plt.scatter(state[1], state[0], s=100, c='b')
                else:
                    plt.arrow(state[1], state[0] , action[1], action[0], color='r', width=0.15, length_includes_head=True)
        plt.axis('off')
        
    
class Game(Environment):
    """Generic 2-player game. The adversary is part of the environment. The agent is player 1 or player -1."""
    
    Board_Size = None
    
    def __init__(self, adversary_policy='random', player=1, play_first=True):
        if play_first:
            self.first_player = player
        else:
            self.first_player = -player
        super(Game, self).__init__()    
        self.adversary = Agent(self, adversary_policy, player=-player)
        self.player = player
        
    @staticmethod
    def is_game():
        return True
    
    def init_state(self):
        player = self.first_player
        board = None
        return player, board

    def is_terminal(self, state):
        player, board = state
        return bool(self.get_reward(state)) or board.astype(bool).all()
    
    @staticmethod
    def encode(state):
        player, board = state
        state_code = player, tuple(board.ravel())
        return state_code
        
    @classmethod
    def decode(cls, state_code):
        player, board = state_code
        if cls.Board_Size is None:
            state = player, np.array(board)
        else:    
            state = player, np.array(board).reshape(cls.Board_Size)
        return state
    
    def get_available_actions(self, state):
        """Get actions in some state, ignoring the player."""
        actions = []
        return actions
    
    def get_actions(self, state, player=None):
        """Get the actions of the specified player in some state."""
        actions = []
        if not self.is_terminal(state):
            player_, board = state
            if player is None or player == player_:
                actions = self.get_available_actions(state)
            else:
                actions = [None]
        return actions
    
    def get_all_actions(self):
        """Get all actions."""
        state = self.init_state()
        actions = self.get_available_actions(state)
        actions.append(None) # action when passing.
        return actions        
    
    @staticmethod
    def get_next_state(state, action):
        return state
    
    def get_transition(self, state, action):
        probs = []
        states = []
        if not self.is_terminal(state):
            player, board = state
            if action is None and player == -self.player:
                probs, actions = self.adversary.policy(state)
            else:
                probs = [1]
                actions = [action]
            for action in actions:
                next_state = self.get_next_state(state, action)
                states.append(next_state)
        return probs, states
    
    def step(self, action=None):
        reward = 0
        stop = True
        state = self.state
        if not self.is_terminal(state):
            player, board = state
            if action is None and player == -self.player:
                action = self.adversary.get_action(state)
            if action is not None:
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(next_state)
                stop = self.is_terminal(next_state)
                self.state = next_state
        return reward, stop
    
        
class TicTacToe(Game):
    """Tic-tac-toe game."""
    
    Board_Size = (3, 3)

    def init_state(self):
        board = np.zeros(TicTacToe.Board_Size).astype(int)
        return (self.first_player, board)
    
    @staticmethod
    def one_hot_encode(state):
        player, board = state
        code = np.hstack((board.ravel()==player, board.ravel()==-player))
        return code
    
    def is_valid(self, state):
        player, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        if 3 in sums and -3 in sums:
            return False
        if player == self.first_player:
            return np.sum(board==player) == np.sum(board==-player)
        else:
            return np.sum(board==player) == np.sum(board==-player) - 1
        
    def get_all_states(self):
        boards = [np.array(board).reshape(TicTacToe.Board_Size) for board in itertools.product([-1, 0, 1], repeat=9)]
        states = [(1, board) for board in boards] + [(-1, board) for board in boards]
        states = [state for state in states if self.is_valid(state)]
        return states
        
    def get_available_actions(self, state):
        """Get available actions in some state."""
        actions = []
        if not self.is_terminal(state):
            _, board = state
            x_, y_ = np.where(board == 0)
            actions = [(x, y) for x, y in zip(x_, y_)]
        return actions
    
    @staticmethod
    def get_reward(state):
        _, board = state
        sums = list(board.sum(axis=0)) + list(board.sum(axis=1))
        sums.append(board.diagonal().sum())
        sums.append(np.fliplr(board).diagonal().sum())
        if 3 in sums:
            reward = 1
        elif -3 in sums:
            reward = -1
        else:
            reward = 0
        return reward
    
    @staticmethod
    def get_next_state(state, action):
        player, board = deepcopy(state)
        board[action] = player
        return -player, board

    def display(self, states=None, marker1='X', marker2='o', marker_size=2000, color1='b', color2='r', interval=300):
        image = 200 * np.ones((3, 3, 3)).astype(int)
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class Nim(Game):
    """Nim game. The player taking the last object looses."""

    Board_Size = [1, 3, 5, 7]

    @classmethod
    def set_parameters(cls, board):
        cls.Board_Size = board

    def init_state(self):
        board = np.array(Nim.Board_Size).astype(int)
        state = (self.first_player, board)
        return state
    
    @staticmethod
    def one_hot_encode(state):
        _, board = state
        n = len(Nim.Board_Size)
        count = np.sum(Nim.Board_Size)
        code = np.zeros(count + n, dtype=bool)
        code[board + np.arange(n)] = 1
        return code
    
    def is_valid(self, state):
        player, board = state
        if player == self.first_player:
            return np.sum(board) != np.sum(Nim.Board_Size) - 1
        else:
            return np.sum(board) != np.sum(Nim.Board_Size)
        
    def get_all_states(self):
        boards = [np.array(board) for board in itertools.product(*(np.arange(k + 1) for k in Nim.Board_Size))]
        states = [(1, board) for board in boards] + [(-1, board) for board in boards]
        states = [state for state in states if self.is_valid(state)]
        return states
    
    def get_available_actions(self, state):
        """Get available actions in some state."""
        actions = []
        if not self.is_terminal(state):
            _, board = state
            rows = np.flatnonzero(board)
            actions = [(row, number + 1) for row in rows for number in range(board[row])]
        return actions

    @staticmethod
    def get_reward(state):
        player, board = state
        if np.sum(board) > 0:
            reward = 0
        else:
            reward = player
        return reward

    @staticmethod
    def is_terminal(state):
        _, board = state
        return not np.sum(board)
    
    @staticmethod
    def get_next_state(state, action):
        player, board = deepcopy(state)
        row, number = action
        board[row] -= number
        return -player, board    

    def display(self, states=None, marker='d', marker_size=500, color_dict={1: 'gold', -1: 'r'}, interval=200):
        board = np.array(Nim.Board_Size).astype(int)
        image = np.zeros((len(board), np.max(board), 3)).astype(int)
        image[:, :, 1] = 135
        if states is not None:
            position = None
            positions = []
            marker_color = []
            for player, board in states:
                x = []
                y = []
                for row in np.where(board)[0]:
                    for col in range(board[row]):
                        x.append(row)
                        y.append(col)
                positions.append((x, y))
                marker_color.append(color_dict[player])
        else:
            positions = None
            player, board = self.state
            x = []
            y = []
            for row in np.where(board)[0]:
                for col in range(board[row]):
                    x.append(row)
                    y.append(col)
            position = x, y
            marker_color = color_dict[player]
        return display_position(image, position, positions, marker, marker_size, marker_color, interval)

    
class ConnectFour(Game):
    """Connect Four game."""
    
    Board_Size = (6, 7)

    def init_state(self):
        board = np.zeros(ConnectFour.Board_Size).astype(int)
        state = (self.first_player, board)
        return state
        
    @staticmethod
    def one_hot_encode(state):
        player, board = state
        code = np.hstack((board.ravel()==player, board.ravel()==-player))
        return code
    
    def get_available_actions(self, state):
        """Get available actions in some state."""
        actions = []
        if not self.is_terminal(state):
            _, board = state
            actions = np.argwhere(board[0] == 0).ravel()
        return actions
    
    @staticmethod
    def get_reward(state):
        _, board = state
        sep = ','
        sequence = np.array2string(board, separator=sep)
        sequence += np.array2string(board.T, separator=sep)
        sequence += ''.join([np.array2string(board.diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        sequence += ''.join([np.array2string(np.fliplr(board).diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        pattern_pos = sep.join(4 * [' 1'])
        pattern_pos_ = sep.join(4 * ['1'])
        pattern_neg = sep.join(4 * ['-1'])
        if pattern_pos in sequence or pattern_pos_ in sequence:
            reward = 1
        elif pattern_neg in sequence:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def get_next_state(state, action):
        player, board = deepcopy(state)
        row = 5 - np.sum(np.abs(board[:, action]))
        board[row, action] = player
        return -player, board
    
    def display(self, states=None, marker1='o', marker2='o', marker_size=1000, colors=['gold', 'r'], interval=200):
        image = np.zeros((*ConnectFour.Board_Size, 3)).astype(int)
        image[:, :, 2] = 255
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, colors[0], colors[1], interval)

    
class FiveInRow(Game):
    """Five-in-a-row game."""

    Board_Size = (10, 10)

    @classmethod
    def init_size(cls, size):
        cls.Board_Size = size

    def init_state(self):
        board = np.zeros(FiveInRow.Board_Size).astype(int)
        state = (self.first_player, board)
        return state
        
    @staticmethod
    def one_hot_encode(state):
        player, board = state
        code = np.hstack((board.ravel()==player, board.ravel()==-player))
        return code
    
    def get_actions(self, state, player=None):
        actions = []
        if not self.is_terminal(state):
            _, board = state
            x_, y_ = np.where(board == 0)
            actions = [(x, y) for x, y in zip(x_, y_)]
        return actions
    
    @staticmethod
    def get_reward(state):
        _, board = state
        sep = ','
        sequence = np.array2string(board, separator=sep)
        sequence += np.array2string(board.T, separator=sep)
        sequence += ''.join([np.array2string(board.diagonal(offset=k), separator=sep) for k in range(-5, 6)])
        sequence += ''.join([np.array2string(np.fliplr(board).diagonal(offset=k), separator=sep) for k in range(-5, 6)])
        pattern_pos = sep.join(5 * [' 1'])
        pattern_pos_ = sep.join(5 * ['1'])
        pattern_neg = sep.join(5 * ['-1'])
        if pattern_pos in sequence or pattern_pos_ in sequence:
            reward = 1
        elif pattern_neg in sequence:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def get_next_state(state, action):
        player, board = deepcopy(state)
        board[action] = player
        return -player, board
    
    def display(self, states=None, marker1='x', marker2='o', marker_size=100, colors=['b', 'r'], interval=200):
        image = 230 * np.ones((*FiveInRow.Board_Size, 3)).astype(int)
        if states is not None:
            boards = [board for _, board in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, colors[0], colors[1], interval)


