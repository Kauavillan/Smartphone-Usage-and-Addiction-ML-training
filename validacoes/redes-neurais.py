# Classificador
from sklearn.neural_network import MLPClassifier

# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

kfold = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state = 1)
classe_array = np.asarray(classe).ravel()
acuracias = []
matrizes = []
metricas = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    classe_array):
    #Melhor configuração do modelo
    classificador = MLPClassifier(
                       verbose=True,
                        max_iter=1500,
                        tol=0.000001,
                        solver="adam",
                        hidden_layer_sizes=[4],
                        activation="tanh",
                        random_state=1,

                   )

    #  Treinamento
    classificador.fit(previsores.iloc[indice_treinamento], classe_array[indice_treinamento])
    
    # teste
    previsoes = classificador.predict(previsores.iloc[indice_teste])
    
    acuracia = accuracy_score(classe_array[indice_teste], previsoes)
    
    metricas.append(precision_recall_fscore_support(classe_array[indice_teste], previsoes))
    matrizes.append(confusion_matrix(classe_array[indice_teste], previsoes))
    acuracias.append(acuracia)
    
##################3 Resultado Final ####################
#Matriz de confusão média
matriz_media = np.mean(matrizes, axis = 0)
matriz_desvio_padrao = np.std(matrizes, axis = 0)
#Métricas médias
acuracias = np.asarray(acuracias)
acuracia_final_media = acuracias.mean()
acuracia_final_desvio_padrao =acuracias.std()
metricas_medias = np.mean(metricas, axis = 0)
metricas_desvio_padrao = np.std(metricas, axis=0)
