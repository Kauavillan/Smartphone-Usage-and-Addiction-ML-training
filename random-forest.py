####### Configuração do Random Forest ##################
import runpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

acuracias_treinamento = []
acuracias_teste = []

# Testar vários valores de árvores
numero_arvores = range(1, 40)
for n in numero_arvores:
	classificador = RandomForestClassifier(
		n_estimators=n,
		criterion="entropy",
		random_state=0,
		n_jobs=-1,
	)
	classificador.fit(previsores_treinamento, classe_treinamento)
	acuracias_treinamento.append(
		classificador.score(previsores_treinamento, classe_treinamento)
	)
	acuracias_teste.append(classificador.score(previsores_teste, classe_teste))

plt.plot(numero_arvores, acuracias_treinamento, label="acuracia de treinamento")
plt.plot(numero_arvores, acuracias_teste, label="acuracia de teste")
plt.ylabel("Acuracia")
plt.xlabel("Numero de arvores")
plt.legend()
plt.show()


############## Classificação com Random Forest ##########

classificador = RandomForestClassifier(
	n_estimators=20,
	criterion="entropy",
	random_state=0,
	n_jobs=-1,
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
plt.xlabel("Importancia da caracteristica")
plt.ylabel("Caracteristica")
plt.title("Importancia das variaveis no Random Forest")
plt.tight_layout()
plt.show()
