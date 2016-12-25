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
os.chdir('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model')

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

def edge_count_to_df(log, num_trials=1, state_id=1, dir_path='sim_01'):
    """Reads nxsim log files and returns the edge count for the topology at every
    time interval of the simulation for every run of the simulation as a pandas
    DataFrame object."""
    D = {}
    for i in range(num_trials):
        name = 'Trial ' + str(i)
        trial = log.open_trial_state_history(dir_path=dir_path, 
                                             trial_id=i)
        edge_count = [len(trial[key]['topology']) for key in trial]
        D[name] = edge_count
    return pd.DataFrame(D)

def graph_measurses_df(log, func, num_trials=1, state_id=1, dir_path='sim_01'):
    """Reads nxsim log files and returns the provided network measure for the
    topology at every time interval of the simulation for every run of the
    simulation as a pandas DataFrame object."""
    D = {}
    for i in range(num_trials):
        name = 'Trial ' + str(i)
        trial = log.open_trial_state_history(dir_path=dir_path, 
                                             trial_id=i)
        measure = [func(nx.Graph(trial[key]['topology'])) 
                   for key in trial]
        D[name] = measure 
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
        self.run_prob = 0.2

    def run(self):
        while True:
            if self.state['id'] == 0:
                self.run_you_fools()
                self.check_for_infection()
                self.count_friends()
                self.get_net()
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def run_you_fools(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.run_prob:
                self.global_topology.remove_edge(self.id, neighbor.id)
                self.state['removed'] = (self.id, neighbor.id)
                print('Rejection:', self.env.now, 'Edge:', self.id, neighbor.id, sep='\t')
    
    def check_for_infection(self):
        zombie_neighbors = self.get_neighboring_agents(state_id=1)
        for neighbor in zombie_neighbors:
            if random.random() < self.inf_prob:
                self.state['id'] = 1 # zombie
                print('Infection:', self.env.now, self.id, '<--', neighbor.id, sep='\t')
                break
                             
    def count_friends(self):
        human_neighbors = self.get_neighboring_agents(state_id=0)
        self.state['friends'] = len(human_neighbors)
        
    def get_net(self):
        nodes = self.get_neighboring_nodes()
        self.state['topology'] = nodes




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
                        max_time=28, num_trials=100, logging_interval=1.0, dir_path='sim_04')

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

plt.title('Escape ZombieApokalypse, $P_{inf} = 0.3$, $P_{run} = 0.2$')
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
friends_01 = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_01').T
escape_friends = friends_to_the_end(BaseLoggingAgent, 50, dir_path='sim_02').T
sentimentality_friends = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_03').T
stigma_friends = friends_to_the_end(BaseLoggingAgent, 100, dir_path='sim_04').T

plt.plot(friends_01.mean(), color='y')
plt.plot(escape_friends.mean(), color='b')
#plt.plot(sentimentality_friends.mean(), color='g')
#lt.plot(stigma_friends.mean(), color='r')

plt.title('Remaining Friends (Humans)')
plt.legend(['ZombieOutBreak','Escape'], loc='best', frameon=True)
#plt.legend(['Escape', 'Sentimentality', 'Stigma'], loc='best', frameon=True)
plt.xlim(xmax=27)
plt.xticks(np.arange(0, 28., 3), tuple(range(1, 29, 3)))
plt.ylabel('Average number of friends')
plt.xlabel('Simulation time')
plt.show()



######################################
#Plotando as redes de infectados

pos=nx.fruchterman_reingold_layout(G)
for i in range(28): 
    
    cor = []
    for j in range(number_of_nodes):
        cor.append(BaseLoggingAgent.open_trial_state_history(dir_path="sim_01", trial_id=1)[i][j]['id'])
    
    cores = []
    for j in cor:
        if j == 1:
            cores.append('red')
        else:
            cores.append('lightgreen')
    
    #plotando a rede
    #plt.figure(i, figsize=(12, 8))
    plt.axis('off')
    nx.draw_networkx_nodes(G,pos,node_size=60,node_color=cores)
    nx.draw_networkx_edges(G,pos,alpha=.4)
    plt.title('Infection - Time '+str(i), size=16)
    plt.show()    
    #if i < 10:
    #    plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras/image00'+str(i+1)+'.png')
    #elif i >= 10:
    #    plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras/image0'+str(i+1)+'.png')


# Extraindo a rede
grafos = []
log = BaseLoggingAgent.open_trial_state_history(dir_path='sim_02', trial_id=6)
for time in range(28):
    print('Grafo no tempo '+str(time))
    e = []
    for node in range(100):
        try:        
            vizinhos = log[time][node]['topology'] 
            for cada in vizinhos:
                edge = (node, cada)
                e.append(edge)
        except KeyError as err:
            print('Erro '+str(node))
    
    grafo = nx.Graph(e)
    grafos.append(grafo)




#Tentando plotar os grafos com edges diferentes na mão
removidos = {}
for time in range(28):
    print('Grafo no tempo '+str(time))
    remov = []
    for node in range(100):
        try:
            rem = log[time][node]['removed']
            remov.append(rem)
            print(rem)
        except KeyError:
            print('Sem removidos na rodada')
            continue
    removidos[time] = remov

pos = nx.fruchterman_reingold_layout(G)
G1 = nx.Graph(G)
G1.remove_edges_from(removidos[1])
nx.draw(G, pos)
#nx.draw(G1)

len(G.edges())
len(G1.edges())

G2 = nx.Graph(G)
G2.remove_edges_from(removidos[2])
#nx.draw(G2)

G3 = nx.Graph(G)
G3.remove_edges_from(removidos[3])
#nx.draw(G3)

G4 = nx.Graph(G)
G4.remove_edges_from(removidos[4])
#nx.draw(G4)

G5 = nx.Graph(G)
G5.remove_edges_from(removidos[5])
#nx.draw(G5)

G6 = nx.Graph(G)
G6.remove_edges_from(removidos[6])
#nx.draw(G6)

G7 = nx.Graph(G)
G7.remove_edges_from(removidos[7])
#nx.draw(G7)

G8 = nx.Graph(G)
G8.remove_edges_from(removidos[8])
#nx.draw(G8)

G9 = nx.Graph(G)
G9.remove_edges_from(removidos[9])
#nx.draw(G9)

G10 = nx.Graph(G)
G10.remove_edges_from(removidos[10])
#nx.draw(G10)

G11 = nx.Graph(G)
G11.remove_edges_from(removidos[11])
#nx.draw(G11)

G12 = nx.Graph(G)
G12.remove_edges_from(removidos[12])
#nx.draw(G12)

G13 = nx.Graph(G)
G13.remove_edges_from(removidos[13])
#nx.draw(G13)

G14 = nx.Graph(G)
G14.remove_edges_from(removidos[14])
#nx.draw(G14)

G15 = nx.Graph(G)
G15.remove_edges_from(removidos[15])
#nx.draw(G15)

G16 = nx.Graph(G)
G16.remove_edges_from(removidos[16])
#nx.draw(G16)

G17 = nx.Graph(G)
G17.remove_edges_from(removidos[17])
#nx.draw(G17)

G18 = nx.Graph(G)
G18.remove_edges_from(removidos[18])
#nx.draw(G18)

G19 = nx.Graph(G)
G19.remove_edges_from(removidos[19])
#nx.draw(G19)

G20 = nx.Graph(G)
G20.remove_edges_from(removidos[20])
#nx.draw(G20)

G21 = nx.Graph(G)
G21.remove_edges_from(removidos[21])
#nx.draw(G21)

G22 = nx.Graph(G)
G22.remove_edges_from(removidos[22])
#nx.draw(G22)

G23 = nx.Graph(G)
G23.remove_edges_from(removidos[23])
#nx.draw(G23)

G24 = nx.Graph(G)
G24.remove_edges_from(removidos[24])
#nx.draw(G24)

G25 = nx.Graph(G)
G25.remove_edges_from(removidos[25])
#nx.draw(G25)

G26 = nx.Graph(G)
G26.remove_edges_from(removidos[26])
#nx.draw(G26)

G27 = nx.Graph(G)
G27.remove_edges_from(removidos[27])
#nx.draw(G27)

grafos = [G,G1,G2,G3,G4,G5,G6,G7,G8,G9,G10,G11,G12,G13,G14,G15,G16,G17,G18,G19,
          G20,G21,G22,G23,G24,G25,G26,G27]

for time in range(28): 
    cor = []
    for j in range(number_of_nodes):
        cor.append(log[time][j]['id'])
    
    cores = []
    for j in cor:
        if j == 1:
            cores.append('red')
        else:
            cores.append('lightgreen')
    
    #plotando a rede
    #plt.figure(i, figsize=(12, 8))
    plt.axis('off')
    pos = nx.fruchterman_reingold_layout(grafos[time])
    nx.draw_networkx_nodes(grafos[time],pos,node_size=60,node_color=cores)
    nx.draw_networkx_edges(grafos[time],pos,alpha=.4)
    #nx.draw_networkx_labels(grafos[time],pos,alpha=.7)
    plt.title('Infection - Time '+str(time), size=16)
    dens = nx.density(grafos[time])
    plt.suptitle('Densidade = '+str(dens))
    
    #if time < 9:
    #    plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras_escape/image00'+str(time+1)+'.png')
    #elif time >= 9:
    #    plt.savefig('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model/figuras_escape/image0'+str(time+1)+'.png')
    plt.show()

    
    
    
densidades = []
for time in range(28):
    dens = nx.density(grafos[time])
    densidades.append(dens)

plt.plot(densidades, color='orange')
plt.title('Escape ZombieApokalypse, $P_{inf} = 0.3$, $P_{run} = 0.2$')
plt.xlim(xmax=27)
plt.xlabel('Time')
plt.ylabel('Density')
plt.show()
#Plotando a densidade


import gc
gc.collect()
