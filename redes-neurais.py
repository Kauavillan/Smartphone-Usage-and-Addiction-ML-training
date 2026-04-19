####### Configuração das Redes Neurais ##################
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


################## Classificação com Redes Neurais ##################

# Treinamento
classificador = MLPClassifier(
	verbose=True,
	max_iter=1000,
	tol=0.000001,
	solver="sgd",
	hidden_layer_sizes=[4],
	activation="relu",
	random_state=1,
)
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e matriz de confusão)
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)

# Resultados na base de treinamento, para verificar overfitting
previsoes_treinamento = classificador.predict(previsores_treinamento)
acuracia_treinamento = accuracy_score(classe_treinamento, previsoes_treinamento)
matriz_treinamento = confusion_matrix(classe_treinamento, previsoes_treinamento)

# Plotagem do "Mapa de calor" dos pesos da primeira camada
plt.imshow(classificador.coefs_[0], interpolation="none", cmap="viridis")
plt.yticks(range(len(cols_previsores)), cols_previsores)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
