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
        self.critica_rate = 3
    
    def run(self):
        while True:
            self.ir_ao_concerto()
            self.ler_criticas()
            self.conversar_com_amigos()
            yield self.env.timeout(1)
            
    def ler_criticas(self):
        if np.random.random() < self.prob_ler_critica:
            print(str(self.id) + ' está lendo as críticas do concerto')
            final_rate = (self.state['quali-rate']+self.critica_rate)/2
            diferenca = final_rate - self.state['quali-rate']
            if self.critica_rate < self.state['quali-rate']:
                print('Diminuiu.', 'Rate:', diferenca, sep='\t')
            else:
                print('Aumentou.', 'Rate:', diferenca, sep='\t')
            #Guarda o final rate
            self.state['quali-rate'] = final_rate
        else:
            print('Não leu críticas.')
            
    def conversar_com_amigos(self):
        for vizinho in self.get_neighboring_agents():
            if np.random.random() < self.prob_concordar:
                meu_rate = self.state['quali-rate']
                vizinho_rate = self.state['quali-rate']
                final_rate = (meu_rate+vizinho_rate)/2
                self.state['quali-rate'] = final_rate
                vizinho.state['quali-rate'] = final_rate
                print('Interação', self.id, vizinho.id, sep='\t')
            else:
                continue
            
    def ir_ao_concerto(self):
        concerto_rate = np.random.normal(3,1,1)
        if np.random.random() < self.prob_ir_ao_concerto:
            final_rate = (concerto_rate + self.state['quali-rate'])/2
                
            if concerto_rate > self.state['quali-rate']:
                print('O concerto foi bom para', self.id, sep=' ')
            else:
                print('O concerto foi ruim para', self.id, sep=' ')
            
            self.state['concerto_rate'] = concerto_rate
            self.state['quali-rate'] = final_rate
        else:
            print(self.id, 'não foi ao concerto.', sep=' ')
        
        
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

#plt.hist(initial_states, bins=5)

init = [{'quali-rate': state} for state in initial_states]

# Preparando a simulação
sim = NetworkSimulation(topology=G, states=init, agent_type=DifusaoQualidade, 
                        max_time=20, num_trials=1, logging_interval=1.0, dir_path='quali_sim_01')

# Rodando a simulação
sim.run_simulation()


# Resultados
log = BaseLoggingAgent.open_trial_state_history(dir_path='quali_sim_01')

rates = []
for node in range(100):
    rate = log[5][node]['quali-rate']
    rates.append(rate)
        
rates = pd.DataFrame(rates)
rates.describe()

mean_rates_no_tempo = []
for time in range(20):
    rates = []
    for node in range(numero_de_pessoas):
        rate = float(log[time][node]['quali-rate'])
        rates.append(rate)
    mean_rate = pd.DataFrame(rates).mean()
    mean_rates_no_tempo.append(mean_rate)

mean_rates_no_tempo = pd.DataFrame(mean_rates_no_tempo)

plt.plot(mean_rates_no_tempo)
plt.title('Rate médio no tempo\t$Críticas = 3$', size=16)
plt.xlabel('Tempo')
plt.ylabel('Rate médio')
plt.ylim(1,5)
plt.show()
