# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Conjunto de dados do Centro de Serviço de Transfusão de Sangue
na cidade de Hsin-Chu em Taiwan - este é um problema de classificação.

### Informação de Atributo
* V1: Tempo para retorno - meses desde a última doação
* V2: Frequência - número total de doações
* V3: Monetário - total de sangue doado em c.c.
* V4: Tempo - meses desde a primeira doação

A classe de saída é uma variável binária que representa se ele doou
sangue em março de 2007 (2 significa doar sangue; 1 significa não doar sangue).

TOTAL DE REGISTROS: 748
570 SÃO CLASSIFICADOS COMO 2
178 SÃO CLASSIFICADOS COMO 1

"""

# BASE DE DADOS - 748 REGISTROS
base_de_dados = pd.read_csv('./blood-transfusion.csv')

# BASE PARA TREINAMENTO 70%
base_de_treinamento = np.array(base_de_dados[:522].drop(columns=['Class']))
# SAÍDA DO TREINAMENTO
saida_treinamento = np.array(base_de_dados[:522]['Class'].values)
saida_treinamento = np.reshape(saida_treinamento, [len(saida_treinamento), 1])


# BASE PARA TESTE 30%
base_de_teste = np.array(base_de_dados[523:].drop(columns=['Class']))
# SAÍDA DO TESTE
saida_teste = np.array(base_de_dados[523:]['Class'].values) 
saida_teste = np.reshape(saida_teste, [len(saida_teste), 1])


# ARRAY QUE CONTÉM AS MÉDIAS ABSOLUTAS DOS ERROS - USADO PARA PLOTAR NO GRÁFICO
erro = []

# INPUT DO NÚMERO DE NEURÔNIOS
num_neuronios = int(input("Informe o número de neurônios: "))

# DEFINIÇÃO DOS PESOS
pesos_entrada = 2 * np.random.random((4, num_neuronios)) - 1
pesos_camada_oculta = 2 * np.random.random((num_neuronios, 1)) - 1

epocas = 1000
taxa_aprendizado = 0.001
momento = 1


''' FUNÇÃO DE ATIVAÇÃO '''
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

''' DERIVADA USADA NA RETROPROPAGAÇÃO '''
def derivada_sigmoid(sig):
    return sig * (1 - sig)


print("TREINANDO...")

# LAÇO PARA TREINAMENTO
for j in range(epocas):
    
    # faz uma cópia --- camada de entrada
    camada_entrada = base_de_treinamento
    
    # multiplica os valores de entrada pelos pesos, em seguida soma tudo (É uma multiplicação de matrizes).
    soma_sinapse_entrada = np.dot(camada_entrada, pesos_entrada)

    # função de ativação
    ativacao_camada_oculta = sigmoid(soma_sinapse_entrada) # esses valores são as entradass da camada oculta.
    
    soma_sinapse_oculta = np.dot(ativacao_camada_oculta, pesos_camada_oculta) # estipula a soma
    ativacao_saida = sigmoid(soma_sinapse_oculta) # função de ativação / saída calculada 
    
    erro_saida = saida_treinamento - ativacao_saida # verifica o erro das saidas calculadas

    # calcula a media do erro com a mean(). A função abs() [absoluta] desconsidera o sinal negativo.
    media_abs = np.mean(np.abs(erro_saida)) 

    erro = np.append(erro, media_abs) # ADICIONANDO OS VALORES DA MÉDIA DO ERRO NO ARRAY DE ERRO PARA PLOTAR

    #print("Media do erro: " + str(media_abs))

    # RETROPROPAGAÇÃO

    delta_saida = erro_saida * derivada_sigmoid(ativacao_saida) # calcula os valores de delta da camada de saida
    delta_oculta = delta_saida.dot(pesos_camada_oculta.T) * derivada_sigmoid(ativacao_camada_oculta) # calc. delta camada oculta.
    
    # ajuste dos pesos
    # formula completa (atualizacao dos pesos camada oculta)
    pesos_camada_oculta = (pesos_camada_oculta * momento) + (ativacao_camada_oculta.T.dot(delta_saida) * taxa_aprendizado)
    pesos_entrada = (pesos_entrada * momento) + (camada_entrada.T.dot(delta_oculta) * taxa_aprendizado)


print("TREINAMENTO FINALIZADO!")

# PLOTAR GRÁFICO DE LINHA DO ERRO
fig, ax = plt.subplots()
ax.plot(erro)
ax.set_title("Média dos Erros")
ax.grid(True)
ax.set_xlabel("Epocas")
#ax.set_ylabel("")
ax.legend()
plt.show()

# FAZER OS TESTES...
'''
soma = np.dot(base_de_teste, pesos_entrada) # multiplica os valores de entrada pelos pesos, em seguida soma tudo.
ativacao_camada_oculta = sigmoid(soma) # esses valores são as entradass da camada oculta.
soma_sinapse_oculta = np.dot(ativacao_camada_oculta, pesos_camada_oculta)
saida = sigmoid(soma_sinapse_oculta) # função de ativação / saída calculada


fig, ax = plt.subplots()
ax.plot(saida[:100], label='Saída prevista')
ax.plot(saida_teste[:100], label='Saíde esperada')
ax.set_title("RNA")
ax.grid(True)
ax.set_xlabel("Epocas")
ax.set_ylabel("...")
ax.legend()
plt.show()
'''