####### Configuração da árvore de decisão ##################
import runpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, accuracy_score


# Reutiliza as variáveis geradas no pipeline de pré-processamento
#contexto = runpy.run_path("pre-processamento.py")
#previsores_treinamento = contexto["previsores_treinamento"]
#previsores_teste = contexto["previsores_teste"]
#classe_treinamento = contexto["classe_treinamento"]
#classe_teste = contexto["classe_teste"]


acuracias_treinamento = []
acuracias_teste = []


# Testar vários valores de profundidade
alturas = range(1, 25)
for altura in alturas:
	classificador = DecisionTreeClassifier(
		criterion="entropy",
		max_depth=altura,
		random_state=0,
	)
	classificador.fit(previsores_treinamento, classe_treinamento)
	acuracias_treinamento.append(
		classificador.score(previsores_treinamento, classe_treinamento)
	)
	acuracias_teste.append(classificador.score(previsores_teste, classe_teste))

plt.plot(alturas, acuracias_treinamento, label="acuracia de treinamento")
plt.plot(alturas, acuracias_teste, label="acuracia de teste")
plt.ylabel("Acuracia")
plt.xlabel("Valor de altura")
plt.legend()
plt.show()


############## Classificação com árvores de decisão ##########

classificador = DecisionTreeClassifier(
	criterion="entropy",
	max_depth=3,
	random_state=0,
)

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste
previsoes = classificador.predict(previsores_teste)

# Análise de resultados
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)

# Visualizando a importância das características
n_features = len(cols_previsores)
plt.figure(figsize=(10, 6))
plt.barh(range(n_features), classificador.feature_importances_, align="center")
plt.yticks(np.arange(n_features), cols_previsores)
plt.xlabel("Importância da característica")
plt.ylabel("Característica")
plt.title("Importância das variáveis na árvore de decisão")
plt.tight_layout()
plt.show()
