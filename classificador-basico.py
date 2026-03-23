import runpy
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score


############ Classificação por classe majoritária ############


# Executa o pipeline de pré-processamento e reutiliza os dados já gerados nele, quando necessário
contexto = runpy.run_path("pre-processamento.py")
classe_treinamento = contexto["classe_treinamento"]
classe_teste = contexto["classe_teste"]



# Resultado mínimo: sempre prever a classe majoritária do treino
contagem = classe_treinamento.value_counts()
classe_majoritaria = contagem.idxmax()
previsoes = pd.Series(
	[classe_majoritaria] * classe_teste.size,
	index=classe_teste.index,
)

# Análise de resultados
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)