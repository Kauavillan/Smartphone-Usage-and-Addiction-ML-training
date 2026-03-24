import runpy
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Reutiliza o pipeline de pre-processamento ja existente no projeto
contexto = runpy.run_path("pre-processamento.py")
X_train = contexto["previsores_treinamento"]
X_test = contexto["previsores_teste"]
y_train = np.ravel(contexto["classe_treinamento"])
y_test = np.ravel(contexto["classe_teste"])

metricas = [
    ("euclidean", {"metric": "euclidean"}),
    ("manhattan", {"metric": "manhattan"}),
    ("chebyshev", {"metric": "chebyshev"}),
    ("minkowski_p1_5", {"metric": "minkowski", "p": 1.5}),
    ("minkowski_p3", {"metric": "minkowski", "p": 3}),
    ("cosine", {"metric": "cosine", "algorithm": "brute"}),
]

resultados = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, params in metricas:
    melhor_k = None
    melhor_cv = -1.0
    melhor_std = None

    for k in range(1, 51):
        modelo = KNeighborsClassifier(n_neighbors=k, **params)
        scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        media = float(scores.mean())
        desvio = float(scores.std())

        if media > melhor_cv:
            melhor_cv = media
            melhor_std = desvio
            melhor_k = k

    modelo_final = KNeighborsClassifier(n_neighbors=melhor_k, **params)
    modelo_final.fit(X_train, y_train)
    y_pred = modelo_final.predict(X_test)

    resultados.append(
        {
            "metrica": nome,
            "melhor_k": melhor_k,
            "cv_acc_media": melhor_cv,
            "cv_acc_std": melhor_std,
            "test_acc": float(accuracy_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
        }
    )

ranking = pd.DataFrame(resultados).sort_values(
    ["cv_acc_media", "test_acc", "test_f1"],
    ascending=False,
)

print(ranking.to_string(index=False))

vencedora = ranking.iloc[0]
print("\nMelhor metrica:", vencedora["metrica"])
print("Melhor k:", int(vencedora["melhor_k"]))
print("CV accuracy media:", round(float(vencedora["cv_acc_media"]), 6))
print("Test accuracy:", round(float(vencedora["test_acc"]), 6))
print("Test F1:", round(float(vencedora["test_f1"]), 6))
