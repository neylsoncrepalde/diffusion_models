#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de difussão social - Neylson Crepalde
Análises
Certifique-se de rodar o script difusao_qualidade.py antes
Para rodar com IPython
"""
import scipy.stats as ss
# Análises

#plotando a distribuição de grau das duas redes
args1 = ss.norm.fit(list(nx.degree(G).values()))
x = np.linspace(0,6)
plt.plot(x, ss.norm.pdf(x, loc=args1[0], scale=args1[1]))
plt.hist(list(nx.degree(G).values()), color='y', bins=6, normed=True)
plt.plot()
plt.title('Distribuição de grau\nRede Small-World')
plt.show()

args2 = ss.expon.fit(list(nx.degree(G_scale_free).values()))
fig = plt.figure(figsize=(8,6))
ax  = fig.add_subplot(111)
x = np.linspace(0,70)
plt.plot(x, ss.expon.pdf(x, loc=args2[0], scale=args2[1]), color='r', lw=3)
plt.hist(list(nx.degree(G_scale_free).values()), normed=True, color='g')
plt.title('Distribuição de grau\nRede Scale-Free')
plt.show()
