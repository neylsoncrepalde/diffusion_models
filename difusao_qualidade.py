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
import random
import os
from nxsim import NetworkSimulation, BaseNetworkAgent, BaseLoggingAgent

sns.set_context('notebook')
os.chdir('/home/neylson/Documentos/Neylson Crepalde/Doutorado/diffusion_model')

class DifusaoQualidade(BaseNetworkAgent):
    def __init__(self, environment=None, agent_id=0, state=()):
        super().__init__(environment=environment, agent_id=agent_id, state=state)
        
        self.prob_ler_critica = 0.3
        self.prob_concordar = 0.6
    
    def run(self):
        while True:
            self.ler_criticas()
            self.conversar_com_amigos()
            self.ir_ao_concerto()
            yield self.env.timeout(1)
    
    def ler_criticas():
        #definir a função
        
    def conversar_com_amigos():
        #definir a função
        
    def ir_ao_concerto():
        #definir a função
        
# Topologia
numero_de_pessoas = 100
G = nx.watts_strogatz_graph(numero_de_pessoas, k=5, p=0.15, seed=123)

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
