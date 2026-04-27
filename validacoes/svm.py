# Classificador
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# Divisão dos dados em folds
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np

# Garante formato 1D para o alvo, funcione ele como Series ou DataFrame de 1 coluna.
classe_array = np.asarray(classe).ravel()

kfold = StratifiedKFold(n_splits = 5, 
                        shuffle = True, 
                        random_state = 1)
acuracias = []
matrizes = []
metricas = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    classe_array):
    #Melhor configuração do modelo
    classificador = SVC(kernel="sigmoid", C=0.9, gamma="auto", random_state=1)

    X_treino = previsores.iloc[indice_treinamento]
    X_teste = previsores.iloc[indice_teste]

    # Padroniza por fold para evitar vazamento de dados
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_teste = scaler.transform(X_teste)

    #  Treinamento
    classificador.fit(X_treino, classe_array[indice_treinamento])
    
    # teste
    previsoes = classificador.predict(X_teste)
    
    acuracia = accuracy_score(classe_array[indice_teste], previsoes)
    
    metricas.append(
        precision_recall_fscore_support(
            classe_array[indice_teste],
            previsoes,
            labels=[0, 1],
            zero_division=0,
        )
    )
    matrizes.append(confusion_matrix(classe_array[indice_teste], previsoes, labels=[0, 1]))
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
