# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:35:20 2023

@author: clemm
"""
from algorithms import PolicyIteration, ValueIteration, SARSA, Q_learning, TD0
import numpy as np 
import matplotlib.pyplot as plt 
import gridworld 


def plot_traj(env, policy, p_type, name):

    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        if p_type == "q_policy": 
            a = np.argmax(policy[s,:])
        elif p_type == "pi": 
            a = policy[s]
            
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    fig, ax = plt.subplots(3)
    ax[0].plot(log['t'], log['s'])
    ax[1].plot(log['t'][:-1], log['a'], color = "orange")
    ax[2].plot(log['t'][:-1], log['r'], color = "green")
    #ax[1].legend([ 'a', 'r'])
    ax[2].set_xlabel("Env step")
    ax[0].set_ylabel("State")
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].set_ylabel("Action")
    ax[2].set_ylabel("Reward")

    fig.suptitle("Gridworld: " + str(name) + " Sample Trajectory")
    
    
def main(): 
    env = gridworld.GridWorld(hard_version=False)
    
    # Policy Iteration 
    alg = PolicyIteration(env, theta = 1e-3, discount = .95)
    i, pi_v, v_mean, pi_pi, de = alg.train()

    iters = np.linspace(1,i, i)
    fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    ax[0].plot(iters, v_mean, label = 'Mean V[s]')    
    ax[1].plot(iters, de, color = "orange",  label = r'$\Delta$')
    ax[1].set_xlabel('Policy Evaluation Iterations')
    ax[1].set_ylabel(r'$\Delta$')
    ax[0].set_xticks([])
    ax[0].set_ylabel('Mean V[s]')
    fig.suptitle('Gridworld: Policy Iteration')
    
    plot_traj(env, pi_pi, "pi", "Policy Iteration")
    
    
    #Value Iteration
    
    algo = ValueIteration(env, theta = 1e-3, discount = 0.95)
    i, vi_v, v_mean, VI_pi, de = algo.train()
    iters = np.linspace(1,i, i)
    fig2, ax2 = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    ax2[0].plot(iters, v_mean, label = 'Mean V[s]')
    ax2[1].plot(iters, de,color = "orange",  label = r'$\Delta$')
    ax2[1].set_xlabel('Evaluation Iterations')
    ax2[0].set_xlabel([])
    ax2[0].set_ylabel('Mean V[s]')
    ax2[1].set_ylabel(r'$\Delta$')
    fig2.suptitle('Gridworld: Value Iteration')
    
    plot_traj(env, VI_pi, "pi", "Value Iteration")
    
    
    
    # the different epsilon and alpha values for SARSA and Q-Learning
    E = [0.1, 0.25, 0.5, 0.75, 0.95]
    A = [1e-3, .01, .1, .5]
    iters = np.linspace(0, 100, 100)
    
    #SARSA

    fig3, ax3 = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    for j in range(len(E)): 
        alg = SARSA(env, eps = E[j], alpha = .1, discount = .95)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        ax3[0].plot(iters, r, label = "$\epsilon$ = " + str(E[j]))
        ax3[1].plot(iters, epsilon)
        
    ax3[0].legend()
    ax3[1].set_xlabel("Training Episode")
    ax3[0].set_ylabel("Avg. Reward")
    ax3[1].set_ylabel(r'$\epsilon$')
    ax3[0].set_xticks([])
    fig3.suptitle( 'Gridworld: SARSA '+ r'$\alpha = 0.1$')
    
    fig4 = plt.figure() 
    for w in range(len(A)): 
        alg = SARSA(env, eps = .75, alpha = A[w], discount = .95)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        plt.plot(iters, r, label = r"$\alpha$ = " + str(A[w]))
        
    plt.legend()
    plt.xlabel("Training Episode")
    plt.ylabel("Return")
    plt.title( 'Gridworld: SARSA '+ r'$\epsilon = 0.75$')
    
    
    
    
    alg = SARSA(env, eps = 0.75, alpha = .01, discount = .95)
    
    Q_SARSA, r, i, SARSA_eps = alg.train()
    plot_traj(env, Q_SARSA, "q_policy", "SARSA")
    
    TD_SARSA = TD0(env, policy = Q_SARSA, alpha = .1, discount = 0.95)
    SARSA_value = TD_SARSA.train()
    
    states = np.linspace(0, env.num_states, env.num_states)
    

    #Q-Learning 
    fig6, ax6 = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]})
    
    for j in range(len(E)): 
        alg = Q_learning(env, eps = E[j], alpha = .1, discount = .95)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        ax6[0].plot(iters, r, label = "$\epsilon$ = " + str(E[j]))
        ax6[1].plot(iters, epsilon)
        
    ax6[0].legend()
    ax6[1].set_xlabel("Training Episode")
    ax6[0].set_ylabel("Return")
    ax6[1].set_ylabel(r'$\epsilon$')
    ax6[0].set_xticks([])
    fig6.suptitle( 'Gridworld: Q-Learning '+ r'$\alpha = 0.1$')
    
    Q_alpha_plot = plt.figure()
    
    for w in range(len(A)): 
        alg = Q_learning(env, eps = .75, alpha = A[w], discount = .95)
        Q, r, i, epsilon = alg.train()
        iters = np.linspace(0, i, i)
        
        plt.plot(iters, r, label = r"$\alpha$ = " + str(A[w]))
        
    plt.legend()
    plt.xlabel("Training Episode")
    plt.ylabel("Avg. Reward")
    plt.title( 'Gridworld: Q-Learning '+ r'$\epsilon = 0.75$')
    
    alg = Q_learning(env, eps = .75, alpha = .1, discount = 0.95)
    Q_learn, r, i, Q_eps = alg.train()
    

    plot_traj(env, Q_learn, "q_policy", "Q-Learning")
    
    TD_Q = TD0(env, policy = Q_learn, alpha = .1, discount = 0.95)
    Q_value = TD_Q.train()
    
    states = np.linspace(0, env.num_states, env.num_states)
  
    
    
    #Plotting the policies
    sarsa_action = []
    q_action = []
    pi_action = []
    vi_action = []
    
    for i in range(env.num_states): 
        sarsa_action.append(np.argmax(Q_SARSA[i,:]))
        q_action.append((np.argmax(Q_learn[i,:])))
        pi_action.append(pi_pi[i])
        vi_action.append(VI_pi[i])
        
    policies = plt.figure()
    plt.scatter(states, pi_action, color = 'red', marker = '^', label = "Policy Iteration")
    plt.scatter(states, vi_action, color = 'orange', marker = '+', label = "Value Iteration")
    plt.scatter(states, sarsa_action, color = 'green', marker = '.', label = "SARSA")
    plt.scatter(states, q_action, color = 'blue', marker = '*', label = "Q-Learning")
    plt.legend()
    plt.xlabel('State')
    plt.yticks([0, 1, 2, 3])
    plt.ylabel('Action')
    plt.title('Gridworld: Policies')
    
    #Plotting value functions
    Values = plt.figure()
    plt.scatter(states, pi_v, color = 'red', marker = '^', label = "Policy Iteration")
    plt.scatter(states, vi_v, color = 'orange', marker = '+', label = "Value Iteration")
    plt.scatter(states, SARSA_value, color = 'green', marker = '.', label = "SARSA")
    plt.scatter(states, Q_value, color = 'blue', marker = '*', label = "Q-Learning")
    plt.legend()
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.title("Gridworld: Value Functions")
    
    
    return 
    
    

if __name__ == '__main__':
    
    main()

