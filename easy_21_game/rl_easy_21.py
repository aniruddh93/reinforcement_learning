##
## Easy21 RL Assignment
## Author: Aniruddh Ramrakhyani
##
## Implements : (i)   Easy 21 game simulation environment
##              (ii)  Dealer Strategy
##              (iii) Monte carlo action value function update
##              (iv)  Sarsa-lambda action value function update
##              (v)   Plotting functions and MSE for action value function
##

import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import getopt
import math

    

class Easy21GameEnv_t:
    """Implements the environment for Easy21 Game"""

    def __init__(self):
        """Set game start condition : Both player and dealer draw a black card"""
        self.dealer_sum = abs(self.drawCard())
        self.dealer_initial_card = self.dealer_sum
        self.player_sum = abs(self.drawCard())
        self.player_initial_card = self.player_sum
        #random.seed(a=0)

    def drawCard(self):
        """Samples and returns a card from deck at random. Card value (1-10) is uniformly 
        distributed while the color distribution probability is : red(1/3) & black(2/3)"""
        value = random.randint(1, 10)
        color = random.randint(1, 30)

        if color <= 10:
            value = -value
        return value

    def step(self, player_action):
        """Simulates a forward step in Easy21 game"""
        step_result = {
            "player_sum" : self.player_sum,
            "reward" : 0,
            "terminal" : False,
            }

        if player_action == "hit":
            self.player_sum += self.drawCard()
            step_result["player_sum"] = self.player_sum

            if (self.player_sum < 1) or (self.player_sum > 21):
                step_result["reward"] = -1
                step_result["terminal"] = True

        elif player_action == "stick":
            terminal_state_reached = False
            
            while(not terminal_state_reached):
                self.dealer_sum += self.drawCard()
                
                if self.dealer_sum >= 17:
                    step_result["terminal"] = True

                    if self.dealer_sum > self.player_sum:
                        step_result["reward"] = -1
                    elif self.dealer_sum == self.player_sum:
                        step_result["reward"] = 0
                    else:
                        step_result["reward"] = 1
                    
                    terminal_state_reached = True

        else:
            sys.exit("Illegal action: " + player_action)

        return step_result




class Easy21_Control:
    """Simulates monte-carlo & sarsa-lambda based policy evaluation and control."""

    def __init__(self, print_episode_update, sarsa_lambda_param, update_policy):
        self.action_value_fn = {}
        self.N_not = 100
        self.num_state_visit = {}
        self.num_state_action_visit = {}
        self.print_start_index = 1
        self.episode_id = 0
        self.print_episode_update = print_episode_update
        self.sarsa_lambda_param = sarsa_lambda_param
        self.eligibility_trace = {}
        self.update_policy = update_policy
        

        # Init the action value function
        for dealer_initial_card in range(1, 11, 1):
            self.action_value_fn[dealer_initial_card] = {}
            self.num_state_visit[dealer_initial_card] = {}
            self.num_state_action_visit[dealer_initial_card] ={}
            self.eligibility_trace[dealer_initial_card] = {}

            for player_sum in range(1, 22, 1):
                self.action_value_fn[dealer_initial_card][player_sum] = {}
                self.num_state_visit[dealer_initial_card][player_sum] = 0
                self.num_state_action_visit[dealer_initial_card][player_sum] = {}
                self.eligibility_trace[dealer_initial_card][player_sum]= {}

                for action in ["hit", "stick"]:
                    self.action_value_fn[dealer_initial_card][player_sum][action] = 0
                    self.num_state_action_visit[dealer_initial_card][player_sum][action] = 0
                    self.eligibility_trace[dealer_initial_card][player_sum][action] = 0

    def simulate_episode(self):
        # Create Easy21_env. This selects initial state as part of init.
        game_env = Easy21GameEnv_t()
        terminal_state_reached = False
        episode_reward = 0
        dealer_initial_card = game_env.dealer_initial_card
        player_sum = game_env.player_initial_card
        action_queue = []
        player_sum_queue = []
        reward_queue = []
        self.episode_id += 1
        
        while(not terminal_state_reached):
            action = self.select_action(dealer_initial_card, player_sum)

            action_queue.append(action)
            player_sum_queue.append(player_sum)

            game_state = game_env.step(action)
            episode_reward = game_state["reward"]
            player_sum = game_state["player_sum"]
            terminal_state_reached = game_state["terminal"]
            reward_queue.append(episode_reward)

            if self.update_policy == "sarsa_lambda":
                self.update_sarsa_lambda_action_value_fn(dealer_initial_card = dealer_initial_card, 
                                                         player_sum_queue = player_sum_queue,
                                                         action_queue = action_queue,
                                                         reward_queue = reward_queue,
                                                         episode_finished = terminal_state_reached)

        if self.update_policy == "monte_carlo":        
            self.update_mc_action_value_fn(dealer_initial_card = dealer_initial_card, 
                                           player_sum_queue = player_sum_queue,
                                           action_queue = action_queue,
                                           reward = episode_reward)

        if self.print_episode_update:
            self.print_episode_summary(dealer_initial_card = dealer_initial_card,
                                       player_sum_queue = player_sum_queue,
                                       action_queue = action_queue,
                                       reward = episode_reward)




    def print_episode_summary(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Prints the summary for the episode"""
        
        # Print episode details
        print_str = "e" + str(self.episode_id) + ":"
        print_str += " episode_reward:" + str(reward)
        print(print_str)

        # Print updated action value fn
        for i in range(0, len(action_queue)):
            cur_action = action_queue[i]
            cur_player_sum = player_sum_queue[i]
            q_s_a = self.action_value_fn[dealer_initial_card][cur_player_sum][cur_action]
            print_str = "(dealer_initial_card:" + str(dealer_initial_card) + ", cur_player_sum:" + str(cur_player_sum) 
            print_str += ", cur_action:" + cur_action + ", q:" + str(q_s_a) + ")"
            print(print_str)
        print("")

    
    def select_action(self, dealer_initial_card, player_sum):
        # With prob. (e/2 + (1-e)), select the greedy action
        # select the other action with prob. (e/2).
        # e = N_not / (N_not + N(s))

        greedy_action = "hit"
        non_greedy_action = "stick"
        if (self.action_value_fn[dealer_initial_card][player_sum]["stick"] > self.action_value_fn[dealer_initial_card][player_sum]["hit"]):
            greedy_action = "stick"
            non_greedy_action = "hit"

        e = self.N_not / (self.N_not + self.num_state_visit[dealer_initial_card][player_sum])
        non_greedy_action_prob = e/2
        prob = random.random()
        
        action = greedy_action
        if prob < e/2:
            action = non_greedy_action
        return action



    def step_update_sarsa_lambda(self, delta, dealer_initial_card, player_sum_t, action_t, action_queue, player_sum_queue, 
                                 reward_t, episode_finished):
        """Update the action value function with sarsa lambda policy"""
        
        # Update visit count and eligibility trace
        self.num_state_visit[dealer_initial_card][player_sum_t] += 1
        self.num_state_action_visit[dealer_initial_card][player_sum_t][action_t] += 1
        self.eligibility_trace[dealer_initial_card][player_sum_t][action_t] += 1

        # adaptive learning rate
        alpha = 1 / self.num_state_action_visit[dealer_initial_card][player_sum_t][action_t]

        # Update action value function
        for i in range(0, len(action_queue)):
            player_sum_i = player_sum_queue[i]
            action_i = action_queue[i]
            self.action_value_fn[dealer_initial_card][player_sum_i][action_i] += alpha * self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] * delta
            self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] *= self.sarsa_lambda_param

            if episode_finished:
                # Reset the eligibility trace since we have reached the terminal state of episode.                               
                self.eligibility_trace[dealer_initial_card][player_sum_i][action_i] = 0

        # Print action value fn update for this step of episode.
        if self.print_episode_update:
            self.print_sarsa_lambda_step_update(dealer_initial_card = dealer_initial_card,
                                                player_sum_queue = player_sum_queue,
                                                action_queue = action_queue,
                                                reward = reward_t)



    def run_terminal_state_sarsa_lambda_update(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue):
        """Update action value function when terminal state is reached."""
        player_sum_t = player_sum_queue[len(action_queue)-1]
        action_t = action_queue[len(action_queue)-1]
        reward = reward_queue[len(action_queue)-1]
        delta_t = reward - self.action_value_fn[dealer_initial_card][player_sum_t][action_t]

        self.step_update_sarsa_lambda(delta = delta_t, 
                                      dealer_initial_card = dealer_initial_card, 
                                      player_sum_t = player_sum_t,
                                      action_t = action_t,
                                      action_queue = action_queue,
                                      player_sum_queue = player_sum_queue,
                                      reward_t = reward,
                                      episode_finished = True)        





    def update_sarsa_lambda_action_value_fn(self, dealer_initial_card, player_sum_queue, action_queue, reward_queue, episode_finished):
        """Implements SARSA-Lambda action value function update.
        e(s,a) = lambda*gamma*e(s,a) + 1(s_t = s, a_t = a)
        delta_t = reward_t + gamma*q(s_t+1, a_t+1) - q(s_t, a_t)
        q(s,a) = q(s,a) + alpha*e(s,a)*delta_t.
        Here we assume gamma=1 (undiscounted).
        alpha = 1 / N(s,a)."""

        if(len(action_queue) < 1):
            print("Nothing to update as no action taken by player !!")
            return


        if episode_finished and (len(action_queue) == 1):
            self.run_terminal_state_sarsa_lambda_update(dealer_initial_card, player_sum_queue, action_queue, reward_queue)
        else:
            if(len(action_queue) < 2):
                return
            player_sum_t = player_sum_queue[len(action_queue)-2]
            action_t = action_queue[len(action_queue)-2]
            reward = reward_queue[len(action_queue)-2]
            player_sum_t_plus_1 = player_sum_queue[len(action_queue)-1]
            action_t_plus_1 = action_queue[len(action_queue)-1]
            delta_t = reward + self.action_value_fn[dealer_initial_card][player_sum_t_plus_1][action_t_plus_1] - self.action_value_fn[dealer_initial_card][player_sum_t][action_t]

            self.step_update_sarsa_lambda(delta = delta_t,
                                          dealer_initial_card = dealer_initial_card,
                                          player_sum_t = player_sum_t,
                                          action_t = action_t,
                                          action_queue = action_queue,
                                          player_sum_queue = player_sum_queue,
                                          reward_t = reward,
                                          episode_finished = False)

            if episode_finished:
                self.run_terminal_state_sarsa_lambda_update(dealer_initial_card = dealer_initial_card,
                                                            player_sum_queue = player_sum_queue,
                                                            action_queue = action_queue,
                                                            reward_queue = reward_queue)


    
    def print_sarsa_lambda_step_update(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Pretty print sarsa lambda action value function update for one step."""
        print("step_reward: " + str(reward))
        for i in range(0, len(action_queue)):
            player_sum_i = player_sum_queue[i]
            action_i = action_queue[i]
            eligibility_trace_i = self.eligibility_trace[dealer_initial_card][player_sum_i][action_i]
            action_value_i = self.action_value_fn[dealer_initial_card][player_sum_i][action_i]

            print_str = "e[" + str(dealer_initial_card) + "][" + str(player_sum_i) + "][" + action_i + "]=" + str(eligibility_trace_i)
            print_str += ", q:" + str(action_value_i)
            print(print_str)
        print("-----------------")



    def update_mc_action_value_fn(self, dealer_initial_card, player_sum_queue, action_queue, reward):
        """Implements monte-carlo action value function update. 
        q(s,a) = q(s,a) + alpha*(reward - q(s,a)).
        alpha = 1 / N(s,a)."""

        for i in range(0, len(action_queue)):
            cur_action = action_queue[i]
            cur_player_sum = player_sum_queue[i]
            
            self.num_state_visit[dealer_initial_card][cur_player_sum] += 1
            self.num_state_action_visit[dealer_initial_card][cur_player_sum][cur_action] += 1

            alpha = 1 / self.num_state_action_visit[dealer_initial_card][cur_player_sum][cur_action]
            self.action_value_fn[dealer_initial_card][cur_player_sum][cur_action] += alpha * (reward - self.action_value_fn[dealer_initial_card][cur_player_sum][cur_action])




    def run_control(self, num_episodes):
        """Runs specified control for the game and returns the action-value function."""
        for i in range(0, num_episodes):
            self.simulate_episode()
            #if i % 5 == 0:
                #print("Finished episode: ", str(i), ". Updated action value fn:")
                #self.print_value_function_update()
        return self.action_value_fn



    def get_state_value_fn(self):
        """Returns state value function (as a 2-d numpy array) using a greedy policy from action value function"""
        state_value_fn = np.zeros(shape=(10, 21))
        
        for dealer_initial_card in range(1, 11, 1):
            for player_sum in range(1, 22, 1):
                hit_value = self.action_value_fn[dealer_initial_card][player_sum]["hit"]
                stick_value = self.action_value_fn[dealer_initial_card][player_sum]["stick"]

                state_value_fn[dealer_initial_card-1][player_sum-1] = hit_value
                if stick_value > hit_value:
                    state_value_fn[dealer_initial_card-1][player_sum-1] = stick_value
        return state_value_fn


    def print_value_function_update(self):
        """Pretty prints the state value function. Prints every fifth entry in state value fn."""
        self.print_start_index = self.print_start_index % 5
        sv_fn = self.get_state_value_fn()
        print_str = ""
        
        for dealer_initial_card in range(0, 10, 1):
            for player_sum in range(self.print_start_index, 21, 1):
                print_str = print_str + str(round(sv_fn[dealer_initial_card][player_sum], 2)) +  "<->"
            print(print_str)
            print_str = ""
        self.print_start_index = self.print_start_index + 1


    def plot_state_value_function(self):
        """Plots the state value function."""
        # Create the input data for X & Y axis
        dealer_showing = np.arange(1, 11, 1)
        player_sum = np.arange(1, 22, 1)
        ds_2d, ps_2d = np.meshgrid(dealer_showing, player_sum)
        sv_2d = self.get_state_value_fn()
        sv_2d = sv_2d.transpose()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surface_plot = ax.plot_surface(ds_2d, ps_2d, sv_2d, linewidth=0, antialiased=False)
        ax.set_xlabel("Dealer showing ----->", fontsize=18)
        ax.set_ylabel("Player Sum ----->", fontsize=16)
        ax.set_zlabel("State Value fn ---->", fontsize=16)
        
        plt.show()




def run_policy(policy, print_episode_update, sarsa_lambda_param, num_episodes, plot_value_fn):
    """Runs specified policy for Easy21 game and returns action value function."""

    if not (policy == "monte_carlo" or policy == "sarsa_lambda"):
        print("Invalid policy: " + policy + ". Valid policies: sarsa_lambda, monte_carlo")
        sys.exit(2)

    easy21_rl = Easy21_Control(print_episode_update=print_episode_update,
                               sarsa_lambda_param = sarsa_lambda_param,
                               update_policy=policy)

    action_value_fn = easy21_rl.run_control(num_episodes=num_episodes)

    if plot_value_fn:
        print("Simulation finished !! Plotting state value function!")
        easy21_rl.plot_state_value_function()

    return action_value_fn




def compute_mse(mc_action_value_fn, sarsa_lamda_action_value_fn):
    """Computes mean-squared error between mc and sarsa_lambda action value functions."""
    mse = 0.0
    for dealer_initial_card in range(1, 11, 1):
        for player_sum in range(1, 22, 1):
            for action in ["hit", "stick"]:
                diff = mc_action_value_fn[dealer_initial_card][player_sum][action]
                mse += diff * diff
    mse /= (11*22*2)
    mse = math.sqrt(mse)
    return mse
    




def run_sarsa_lambda_sweep_and_plot_error(print_episode_update, num_episodes):
    """Runs : (i) Monte-carlo value fn update, (ii) sarsa lambda value function update with lambda varying from 0.0 to 1.0 (0.1 increment) &
    (iii) Plots the mean squared error of the value function computed with monte_carlo and sarsa_lambda methods."""

    # for plotting
    x = np.arange(0.0, 1.1, 0.1)
    y = []

    # monte-carlo action value fn
    mc_action_value_fn = run_policy(policy = "monte_carlo",
                                    print_episode_update = print_episode_update, 
                                    sarsa_lambda_param = 0.0,
                                    num_episodes = 100000,
                                    plot_value_fn = False)

    for sarsa_lambda_param in np.arange(0.0, 1.1, 0.1):
        sarsa_lamda_action_value_fn = run_policy(policy = "sarsa_lambda",
                                                 print_episode_update = print_episode_update,
                                                 sarsa_lambda_param = sarsa_lambda_param,
                                                 num_episodes = num_episodes,
                                                 plot_value_fn = False)

        mse = compute_mse(mc_action_value_fn, sarsa_lamda_action_value_fn)
        print("lambda:" + str(sarsa_lambda_param) + ", mse:" + str(mse))
        y.append(mse)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y)
    ax.set_xlabel("sarsa_lambda param -------->")
    ax.set_ylabel("Mean Squared Error -------->")
    ax.set_title("Plot of Mean squared error (between mc and sl) with lambda sweep")
    plt.show()



def main(argv):
    """Program main"""
  
    # defaults
    policy = "monte_carlo"
    num_episodes = 200000
    print_episode_update = False
    sarsa_lambda_value = 1.0
    run_sarsa_lambda_sweep = False

    try:
        opts, args = getopt.getopt(argv,"hp:",["policy=", "num-episodes=", "print-episode-update=", "sarsa_lambda_value=", 
                                               "run-sarsa-lambda-sweep="])

    except getopt.GetoptError:
        print("abc.py -p <rl update policy: sarsa_lambda, monte_carlo> --num-episodes=<no. of episodes to simulate> --print-episode-update=<True or False>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print("abc.py -p <rl update policy: sarsa_lambda, monte_carlo> --num-episodes=<no. of episodes to simulate> --print-episode-update=<True or False>")
            sys.exit()
        elif opt in ("-p", "--policy"):
            policy = arg
        elif opt in ("--num-episodes"):
            num_episodes = int(arg)
        elif (opt in ("--print-episode-update")) and (arg == "True"):
            print_episode_update = True
        elif (opt in ("--sarsa_lambda_value")):
            sarsa_lambda_value = float(arg)
        elif (opt in ("--run-sarsa-lambda-sweep")) and (arg == "True"):
            run_sarsa_lambda_sweep = True
        


    if run_sarsa_lambda_sweep:
        run_sarsa_lambda_sweep_and_plot_error(print_episode_update, num_episodes)
    else:
        run_policy(policy=policy,
                   print_episode_update=print_episode_update,
                   sarsa_lambda_param = sarsa_lambda_value,
                   num_episodes = num_episodes,
                   plot_value_fn = True)




if __name__ == "__main__":
    main(sys.argv[1:])
