####### Configuração do KNN ##################
import runpy
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# Reutiliza as variáveis geradas no pipeline de pré-processamento
contexto = runpy.run_path("pre-processamento.py")
previsores_treinamento = contexto["previsores_treinamento"]
previsores_teste = contexto["previsores_teste"]
classe_treinamento = contexto["classe_treinamento"]
classe_teste = contexto["classe_teste"]

acuracias_treinamento = []
acuracias_teste = []


# testar varios valores de K
numero_vizinhos = range(1,51)
for k in numero_vizinhos:
    #construir o modelo
    classificador = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    # treinar modelo
    classificador.fit(previsores_treinamento, classe_treinamento)
    acuracias_treinamento.append(classificador.score(previsores_treinamento,classe_treinamento))
    acuracias_teste.append(classificador.score(previsores_teste,classe_teste))

plt.plot(numero_vizinhos, acuracias_treinamento, label="acuracia de treinamento")
plt.plot(numero_vizinhos, acuracias_teste, label="acuracia de teste")
plt.ylabel("Acuracia")
plt.xlabel("Valor de K")
plt.legend()

############## Classificação com KNN ##########

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# minkowski com p=2 é a distância euclidiana
classificador = KNeighborsClassifier(n_neighbors=13, metric='manhattan')

#  Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# teste
previsoes = classificador.predict(previsores_teste)

#análise de resultados
from sklearn.metrics import confusion_matrix, accuracy_score
acuracia_teste = accuracy_score(classe_teste,previsoes)
matriz_teste = confusion_matrix(classe_teste,previsoes)