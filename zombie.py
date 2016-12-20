# -*- coding: utf-8 -*-
'''
Zombie Apocalypse
'''
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import random
import os
from nxsim import NetworkSimulation, BaseNetworkAgent, BaseLoggingAgent

sns.set_context('notebook')

def census_to_df(log, num_trials=1, state_id=1, dir_path='sim_01'):
    """Reads nxsim log files and returns the sum of agents with a given state_id at 
    every time interval of the simulation for every run of the simulation as a pandas
    DataFrame object."""
    D = {}
    for i in range(num_trials):
        name = 'Trial ' + str(i)
        trial = log.open_trial_state_history(dir_path=dir_path, 
                                             trial_id=i)
        census = [sum([1 for node_id, state in g.items() 
                       if node_id != 'topology' and state['id'] == state_id]) 
                  for t, g in trial.items()]
        D[name] = census      
    return pd.DataFrame(D)

def friends_to_the_end(log, num_trials=1, dir_path='sim_01'):
    """Reads nxsim log files and returns the number of human friends connected
    to every human at every time interval of the simulation for every run of
    the simulation as a pandas DataFrame object."""
    D = {}
    for i in range(num_trials):
        name = 'Trial ' + str(i)
        trial = log.open_trial_state_history(dir_path=dir_path, 
                                             trial_id=i)
        friends = [np.mean([state['friends'] for node_id, state in g.items() 
                       if node_id != 'topology' and state['id'] == 0]) 
                  for t, g in trial.items()]
        D[name] = friends    
    return pd.DataFrame(D)    

number_of_nodes = 100
G = nx.scale_free_graph(number_of_nodes).to_undirected()

class ZombieMassiveOutbreak(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.inf_prob = 0.2

    def run(self):
        while True:
            if self.state['id'] == 0:
                self.check_for_infection()
                self.count_friends()
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def check_for_infection(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.inf_prob:
                self.state['id'] = 1
                print(self.env.now, self.id, '<--', neighbor.id, sep='\t')
                break
                                
    def count_friends(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        self.state['friends'] = len(human_neighbors)


class ZombieEscape(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.inf_prob = 0.3
        self.run_prob = 0.05

    def run(self):
        while True:
            if self.state['id'] == 0:
                self.run_you_fools()
                self.check_for_infection()
                self.count_friends()
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def check_for_infection(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.inf_prob:
                self.state['id'] = 1 # zombie
                print('Infection:', self.env.now, self.id, '<--', neighbor.id, sep='\t')
                break
                
    def run_you_fools(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.run_prob:
                self.global_topology.remove_edge(self.id, neighbor.id)
                print('Rejection:', self.env.now, 'Edge:', self.id, neighbor.id, sep='\t')
                
    def count_friends(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        self.state['friends'] = len(human_neighbors)

class ZombieSentimentality(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.inf_prob = 0.3
        self.run_prob = 0.05
        self.sent_coef = 0.01

    def run(self):
        while True:
            if self.state['id'] == 0:
                self.check_for_sentimentality()
                self.check_for_infection()
                self.count_friends()
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def check_for_infection(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.inf_prob:
                self.state['id'] = 1 # zombie
                print('Infection:', self.env.now, self.id, '<--', neighbor.id, sep='\t')
                break
                
    def check_for_sentimentality(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        sent_prob = self.run_prob - (self.env.now * self.sent_coef)
        for neighbor in zombie_neighbors:
            if random.random() < sent_prob:
                self.global_topology.remove_edge(self.id, neighbor.id)
                print('Rejection:', self.env.now, 'Edge:', self.id, neighbor.id, sep='\t')

    def count_friends(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        self.state['friends'] = len(human_neighbors)



class ZombieStigma(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.inf_prob = 0.3
        self.run_prob = 0.05
        self.sent_coef = 0
        self.stigma_coef = 0.02

    def run(self):
        while True:
            if self.state['id'] == 0:
                self.check_for_sentimentality()
                self.stigmatize()
                self.check_for_infection()
                self.count_friends()
                
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def stigmatize(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        for neighbor in human_neighbors:
            sorrounding_zombies = neighbor.get_neighboring_agents(state_id=1)
            if random.random() < (len(sorrounding_zombies) * self.stigma_coef):
                self.global_topology.remove_edge(self.id, neighbor.id)
                print('Stigmatize:', self.env.now, 'Edge:', self.id, neighbor.id, sep='\t')
                
    def check_for_infection(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.inf_prob:
                self.state['id'] = 1 # zombie
                print('Infection:', self.env.now, self.id, '<--', neighbor.id, sep='\t')
                break
                
    def check_for_sentimentality(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        sent_prob = self.run_prob - (self.env.now * self.sent_coef)
        for neighbor in zombie_neighbors:
            if random.random() < sent_prob:
                self.global_topology.remove_edge(self.id, neighbor.id)
                print('Rejection:', self.env.now, 'Edge:', self.id, neighbor.id, sep='\t')

    def count_friends(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        self.state['friends'] = len(human_neighbors)
    
    
    
    
        



# Starting out with a human population
init_states = [{'id': 0, } for _ in range(number_of_nodes)]

# Randomly seeding patient zero
patient_zero = random.randint(0, number_of_nodes)
init_states[patient_zero] = {'id': 1}

# Setting up the simulation
sim = NetworkSimulation(topology=G, states=init_states, agent_type=ZombieMassiveOutbreak, 
                        max_time=28, num_trials=100, logging_interval=1.0, dir_path='sim_01')

# Running the simulation
sim.run_simulation()



# Starting out with a human population
init_states = [{'id': 0, } for _ in range(number_of_nodes)]

# Randomly seeding patient zero
patient_zero = random.randint(0, number_of_nodes)
init_states[patient_zero] = {'id': 1}

# Setting up the simulation
sim = NetworkSimulation(topology=G, states=init_states, agent_type=ZombieEscape, 
                        max_time=28, num_trials=100, logging_interval=1.0, dir_path='sim_02')

# Running the simulation
sim.run_simulation()



# Starting out with a human population
init_states = [{'id': 0, } for _ in range(number_of_nodes)]

# Randomly seeding patient zero
patient_zero = random.randint(0, number_of_nodes)
init_states[patient_zero] = {'id': 1}

# Setting up the simulation
sim = NetworkSimulation(topology=G, states=init_states, agent_type=ZombieSentimentality, 
                        max_time=28, num_trials=100, logging_interval=1.0, dir_path='sim_03')

# Running the simulation
sim.run_simulation()


# Starting out with a human population
init_states = [{'id': 0, } for _ in range(number_of_nodes)]

# Randomly seeding patient zero
patient_zero = random.randint(0, number_of_nodes)
init_states[patient_zero] = {'id': 1}

# Setting up the simulation
sim = NetworkSimulation(topology=G, states=init_states, agent_type=ZombieStigma, 
                        max_time=28, num_trials=1, logging_interval=1.0, dir_path='sim_04')

# Running the simulation
sim.run_simulation()




zombies = census_to_df(BaseLoggingAgent, 100, 1, dir_path='sim_01').T
humans = census_to_df(BaseLoggingAgent, 100, 0, dir_path='sim_01').T

plt.plot(zombies.mean(), color='b')
plt.fill_between(zombies.columns, zombies.max(), zombies.min(), color='b', alpha=.33)

plt.plot(humans.mean(), color='g')
plt.fill_between(humans.columns, humans.max(), humans.min(), color='g', alpha=.33)

plt.title('Simple ZombieApokalypse, $P_{inf}=0.2$')
plt.legend(['Zombies', 'Humans'], loc=7, frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Population')
plt.xlabel('Simulation time')
plt.show()

zombies = census_to_df(BaseLoggingAgent, 100, 1, dir_path='sim_02').T
humans = census_to_df(BaseLoggingAgent, 100, 0, dir_path='sim_02').T

## For later comparisons:
mean_escape_zombies = zombies.mean()
mean_escape_humans = humans.mean()

plt.plot(zombies.mean(), color='b')
plt.fill_between(zombies.columns, zombies.max(), zombies.min(), color='b', alpha=.2)

plt.plot(humans.mean(), color='g')
plt.fill_between(humans.columns, humans.max(), humans.min(), color='g', alpha=.2)

plt.title('Escape ZombieApokalypse, $P_{inf} = 0.3$, $P_{run} = 0.05$')
plt.legend(['Zombies', 'Humans'], loc=7, frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Population')
plt.xlabel('Simulation time')
plt.show()


# plotting sentimentality
zombies = census_to_df(BaseLoggingAgent, 100, 1, dir_path='sim_03').T
humans = census_to_df(BaseLoggingAgent, 100, 0, dir_path='sim_03').T

plt.plot(zombies.mean(), color='r')
plt.plot(humans.mean(), color='y')


plt.plot(mean_escape_zombies, color='b')
plt.plot(mean_escape_humans, color='g')

plt.title('Sentimentality ZombieApokalypse, $P_{inf} = 0.3$, $P_{run} = 0.05$, $P_{sent} = 0.01$')
plt.legend(['"Sentimentality" Zombies', '"Sentimentality" Humans', 
            '"Escape" Zombies', '"Escape" Humans'], 
           loc=7, frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Population')
plt.xlabel('Simulation time')
plt.show()

# Zombie Stigmatization
zombies = census_to_df(BaseLoggingAgent, 100, 1, dir_path='sim_04').T
humans = census_to_df(BaseLoggingAgent, 100, 0, dir_path='sim_04').T

plt.plot(zombies.mean(), color='b')
plt.fill_between(zombies.columns, zombies.max(), zombies.min(), color='b', alpha=.2)

plt.plot(humans.mean(), color='g')
plt.fill_between(humans.columns, humans.max(), humans.min(), color='g', alpha=.2)

plt.title('Stigma ZombieApokalypse, $P_{inf} = 0.3$, $P_{run} = 0.05$, $P_{sti}=0.02$')
plt.legend(['Zombies', 'Humans'], loc=7, frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Population')
plt.xlabel('Simulation time')
plt.show()


# Friends count
escape_friends = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_02').T
sentimentality_friends = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_03').T
stigma_friends = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_04').T

plt.plot(escape_friends.mean(), color='b')
plt.plot(sentimentality_friends.mean(), color='g')
plt.plot(stigma_friends.mean(), color='r')

plt.title('Remaining Friends (Humans)')
plt.legend(['Escape', 'Sentimentality', 'Stigma'], loc='best', frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Average number of friends')
plt.xlabel('Simulation time')
plt.show()

######################################
#Plotando as redes de infectados
os.chdir('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model')
pos=nx.fruchterman_reingold_layout(G)
for i in range(28): 
    
    cor = []
    for j in range(number_of_nodes):
        cor.append(BaseLoggingAgent.open_trial_state_history(dir_path="sim_04", trial_id=5)[i][j]['id'])
    
    cores = []
    for j in cor:
        if j == 1:
            cores.append('red')
        else:
            cores.append('lightgreen')
    
    #plotando a rede
    plt.figure(i, figsize=(16, 12))
    plt.axis('off')
    nx.draw_networkx_nodes(G,pos,node_size=60,node_color=cores)
    nx.draw_networkx_edges(G,pos,alpha=.4)
    plt.title('Infection - Time '+str(i), size=16)
    if i < 10:
        plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras/image00'+str(i+1)+'.png')
    elif i >= 10:
        plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras/image0'+str(i+1)+'.png')

