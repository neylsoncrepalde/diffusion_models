# -*- coding: utf-8 -*-
'''
Network Diffusion Model
Neylson Crepalde
'''
import seaborn as sns
import networkx as nx
from matplotlib import pyplot as plt
sns.set_context('notebook')

number_of_nodes = 20
G = nx.complete_graph(number_of_nodes)

import random
from nxsim import BaseNetworkAgent, BaseLoggingAgent

class ZombieOutbreak(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        self.bite_prob = 0.05

    def run(self):
        while True:
            if self.state['id'] == 1:
                self.zombify()
                yield self.env.timeout(1)
            else:
                yield self.env.event()

    def zombify(self):
        normal_neighbors = self.get_neighboring_agents(state_id=0)
        for neighbor in normal_neighbors:
            if random.random() < self.bite_prob:
                neighbor.state['id'] = 1 # zombie
                print(self.env.now, self.id, neighbor.id, sep='\t')
                break

from nxsim import NetworkSimulation

# Initialize agent states. Let's assume everyone is normal.
# Add keys as as necessary, but "id" must always refer to that state category
init_states = [{'id': 0, } for _ in range(number_of_nodes)]

# Seed a zombie
init_states[5] = {'id': 1}
sim = NetworkSimulation(topology=G, states=init_states, agent_type=ZombieOutbreak,
                        max_time=50, dir_path='sim_01', num_trials=1, logging_interval=1.0)

sim.run_simulation()

#Plotando a rede bonitinha
#plt.figure(1, figsize=(12, 8))             #definindo o tamanho da figura
pos=nx.fruchterman_reingold_layout(G)      #definindo o algoritmo do layout
plt.axis('off')                            #retira as bordas
nx.draw_networkx_nodes(G,pos,node_size=50) #plota os nodes
nx.draw_networkx_edges(G,pos,alpha=0.4)    #plota os ties
plt.title('Initial Network', size=16)     #Título
plt.show()                                 #plota


trial = BaseLoggingAgent.open_trial_state_history(dir_path='sim_01', trial_id=0)


zombie_census = [sum([1 for node_id, state in g.items() if state['id'] == 1]) for t,g in trial.items()]
plt.plot(zombie_census)
plt.title('Infection Model  $P_{inf} = 0.05$', size=14)
plt.show()