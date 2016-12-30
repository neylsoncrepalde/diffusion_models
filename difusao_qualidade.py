# -*- coding: utf-8 -*-
"""
Modelo de Difusão Social
Neylson Crepalde
"""
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import os
from nxsim import NetworkSimulation, BaseNetworkAgent, BaseLoggingAgent

sns.set_context('notebook')
os.chdir('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model')

class DifusaoQualidade(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.prob_ler_critica = 0.3
        self.prob_concordar = 0.6
        self.prob_ir_ao_concerto = 0.5
    
    def run(self):
        while True:
            self.ir_ao_concerto()
            self.ler_criticas()
            self.conversar_com_amigos()
            yield self.env.timeout(1)
    
    def ler_criticas(self):
        for node in self.get_all_nodes():
            if np.random.random() < self.prob_ler_critica:
                print(self.id + ' lendo as críticas do concerto')
                critica_rate = np.random.normal(3,1,1)
                self.state['critica_rate'] = critica_rate
                self.state['quali_rate'] = (self.state['quali_rate']+critica_rate)/2
                if critica_rate < self.state['quali_rate']:
                    print('Diminuiu.', 'Rate:', critica_rate, sep='\t')
                else:
                    print('Aumentou.', 'Rate:', critica_rate, sep='\t')
            else:
                continue
            
    def conversar_com_amigos(self):
        for vizinho in self.get_neighboring_agents():
            
            print('Interação')
            
        
    def ir_ao_concerto(self):
        #definir a função
        if np.random.random() < self.prob_ir_ao_concerto:
            print('Foi ao concerto')
            #continua
        
        
# Topologia
numero_de_pessoas = 100
G = nx.watts_strogatz_graph(numero_de_pessoas, k=4, p=0.15, seed=123)

G_scale_free = nx.scale_free_graph(numero_de_pessoas, seed=123)

#Gerando os states iniciais como uma distribuição normal
initial_states = np.random.normal(3,1,numero_de_pessoas).round(0)

#Colocando no intervalo entre 1 e 5
for i in range(len(initial_states)):
    if initial_states[i] < 1:
        initial_states[i] = 1
    
    elif initial_states[i] > 5:
        initial_states[i] = 5
initial_states

plt.hist(initial_states, bins=5)

init = [{'quali-rate': state} for state in initial_states]
