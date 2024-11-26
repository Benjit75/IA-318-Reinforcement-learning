#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
November 2024
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from collections import defaultdict


class Agent:
    """Agent interacting with some environment.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    """
    
    def __init__(self, model, policy='random', player=None):
        if type(policy) == str:
            if policy == 'random':
                policy = self.random_policy
            elif policy == 'one_step':
                policy = self.one_step_policy
            else:
                raise ValueError("The policy must be either 'random', 'one_step', or a custom policy.")            
        if player is None:
            if model.is_game():
                player = model.player
            else:
                player = 1            
        self.model = model
        self.policy = policy
        self.player = player
            
    def get_actions(self, state):
        """Get available actions."""
        if self.model.is_game():
            return self.model.get_actions(state, self.player)
        else:
            return self.model.get_actions(state)
        
    def random_policy(self, state):
        """Random choice among possible actions."""
        probs = []
        actions = self.get_actions(state)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        return probs, actions
    
    def one_step_policy(self, state):
        """One-step policy for games, looking for win moves or moves avoiding defeat."""
        if not self.model.is_game():
            raise ValueError('The one-step policy is applicable to games only.')
        player, board = state
        if player == self.player:
            actions = self.get_actions(state)
            # win move
            for action in actions:
                next_state = self.model.get_next_state(state, action)
                if self.model.get_reward(next_state) == player:
                    return [1], [action]
            # move to avoid defeat
            for action in actions:
                adversary_state = -player, board
                next_state = self.model.get_next_state(adversary_state, action)
                if self.model.get_reward(next_state) == -player:
                    return [1], [action]
            # otherwise, random move
            if len(actions):
                probs = np.ones(len(actions)) / len(actions)
                return probs, actions
        return [1], [None]

    def get_action(self, state):
        """Get selected action."""
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action      
         
    def get_episode(self, state=None, horizon=100):
        """Get the states and rewards for an episode, starting from some state."""
        self.model.reset(state)
        state = self.model.state
        states = [state] 
        reward = self.model.get_reward(state)
        rewards = [reward]
        stop = self.model.is_terminal(state)
        if not stop:
            for t in range(horizon):
                action = self.get_action(state)
                reward, stop = self.model.step(action)
                state = self.model.state
                states.append(state)
                rewards.append(reward)
                if stop:
                    break
        return stop, states, rewards
    
    def get_gains(self, state=None, horizon=100, n_runs=100, gamma=1):
        """Get the gains (cumulative rewards) over independent runs, starting from some state."""
        gains = []
        for t in range(n_runs):
            _, _, rewards = self.get_episode(state, horizon)
            gains.append(sum(rewards * np.power(gamma, np.arange(len(rewards)))))
        return gains
    
    
class OnlineEvaluation(Agent):
    """Online evaluation. The agent interacts with the environment and learns the value function of its policy.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    gamma : float
        Discount rate (in [0, 1], default = 1).
    init_value : float
        Initial value of the value function.
    """
    
    def __init__(self, model, policy='random', player=None, gamma=1, init_value=0):
        super(OnlineEvaluation, self).__init__(model, policy, player)   
        self.gamma = gamma 
        self.value = defaultdict(lambda: init_value) # value of a state
        self.count = defaultdict(lambda: 0) # count of a state (number of visits)        
            
    def add_state(self, state):
        """Add a state if unknown."""
        code = self.model.encode(state)
        if code not in self.value:
            self.value[code] = 0
            self.count[code] = 0
        
    def get_known_states(self):
        """Get known states."""
        states = [self.model.decode(code) for code in self.value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        code = self.model.encode(state)
        return code in self.value

    def get_values(self, states=None):
        """Get the values of some states (default = all states).""" 
        if states is None:
            states = self.model.get_all_states()
            if not len(states):
                raise ValueError("The state space is too large. Please specify some states.")
        codes = [self.model.encode(state) for state in states]
        values = [self.value[code] for code in codes]
        return values
    
    def get_best_actions(self, state):
        """Get the best actions in some state according to the value function.""" 
        actions = self.get_actions(state)
        if len(actions) > 1:
            values = []
            for action in actions:
                probs, next_states = self.model.get_transition(state, action)
                rewards = [self.model.get_reward(next_state) for next_state in next_states]
                next_values = self.get_values(next_states)
                # expected value
                value = np.sum(np.array(probs) * (np.array(rewards) + self.gamma * np.array(next_values)))
                values.append(value)
            if self.player == 1:
                best_value = max(values)
            else:
                best_value = min(values)
            actions = [action for action, value in zip(actions, values) if value==best_value]
        return actions        
    
    def get_policy(self):
        """Get the best policy according to the value function."""
        def policy(state):
            actions = self.get_best_actions(state)
            if len(actions):
                probs = np.ones(len(actions)) / len(actions)
            else:
                probs = []
            return probs, actions
        return policy    
    
    def randomize_policy(self, epsilon=0):
        """Get the epsilon-greedy policy according to the value function."""
        def policy(state):
            actions = self.get_actions(state)
            best_actions = self.get_best_actions(state)
            probs = []
            for action in actions:
                prob = epsilon / len(actions)
                if action in best_actions:
                    prob += (1 - epsilon) / len(best_actions)
                probs.append(prob)
            return probs, actions
        return policy
     
        
class OnlineControl(OnlineEvaluation):
    """Online control. The agent interacts with the model and learns the best policy.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Initial policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    gamma : float
        Discount rate (in [0, 1], default = 1).
    init_value : float
        Initial value of the action-value function.
    """
    
    def __init__(self, model, policy='random', player=None, gamma=1, init_value=0):
        super(OnlineControl, self).__init__(model, policy, player, gamma)  
        self.action_value = defaultdict(lambda: defaultdict(lambda: init_value))
        self.action_count = defaultdict(lambda: defaultdict(lambda: 0))
            
    def get_known_states(self):
        """Get known states."""
        states = [self.model.decode(code) for code in self.action_value]
        return states
    
    def get_best_actions(self, state):
        """Get the best actions in some state according to the action-value function.""" 
        actions = self.get_actions(state)
        if len(actions) > 1:
            code = self.model.encode(state)
            values = [self.action_value[code][action] for action in actions]
            if self.player == 1:
                best_value = max(values)
            else:
                best_value = min(values)
            actions = [action for action, value in zip(actions, values) if value==best_value]
        return actions
    
    def randomize_best_action(self, state, epsilon=0):
        """Get the action of the epsilon-greedy policy."""
        policy = self.randomize_policy(epsilon=epsilon)
        probs, actions = policy(state)
        i = np.random.choice(len(actions), p=probs)
        return actions[i]