####### Configuração do Naive Bayes ##################
import runpy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score


# Reutiliza as variáveis geradas no pipeline de pré-processamento
#contexto = runpy.run_path("pre-processamento.py")
#previsores_treinamento = contexto["previsores_treinamento"]
#previsores_teste = contexto["previsores_teste"]
#classe_treinamento = contexto["classe_treinamento"]
#classe_teste = contexto["classe_teste"]


################## Classificação com Naive Bayes ##################

classificador = GaussianNB()

# Treinamento
classificador.fit(previsores_treinamento, classe_treinamento)

# Teste
previsoes = classificador.predict(previsores_teste)

# Análise dos resultados (porcentagem de acertos e matriz de confusão)
acuracia_teste = accuracy_score(classe_teste, previsoes)
matriz_teste = confusion_matrix(classe_teste, previsoes)
