import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler


# =============================================================================
#                         Leitura da base e resumo
# =============================================================================

base = pd.read_csv("raw_dataset.csv")
resumo = base.describe(include="all")


# =============================================================================
#                     Tratando valores inválidos e faltantes
# =============================================================================

# Colunas numéricas para tratamento
colunas_numericas = [
	"age",
	"daily_screen_time_hours",
	"social_media_hours",
	"gaming_hours",
	"work_study_hours",
	"sleep_hours",
	"notifications_per_day",
	"app_opens_per_day",
	"weekend_screen_time",
]

# Colunas categóricas para tratamento
colunas_categoricas = [
	"gender",
	"stress_level",
	"academic_work_impact",
	"addiction_level",
]

# Regra simples de validação: valores numéricos negativos são inválidos
for coluna in colunas_numericas:
	base.loc[base[coluna] < 0, coluna] = pd.NA

# Preenche numéricos faltantes com a mediana
for coluna in colunas_numericas:
	mediana = base[coluna].median()
	base.loc[base[coluna].isnull(), coluna] = mediana

# Preenche categóricos faltantes com o valor mais frequente (moda)
for coluna in colunas_categoricas:
	moda = base[coluna].mode(dropna=True)
	if not moda.empty:
		base.loc[base[coluna].isnull(), coluna] = moda.iloc[0]


# =============================================================================
#                     Separando dados em previsores e classe
# =============================================================================

cols_previsores = [
	"age",
	"gender",
	"daily_screen_time_hours",
	"social_media_hours",
	"gaming_hours",
	"work_study_hours",
	"sleep_hours",
	"notifications_per_day",
	"app_opens_per_day",
	"weekend_screen_time",
	"stress_level",
	"academic_work_impact",
	"addiction_level",
]
col_classe = "addicted_label"

previsores = base[cols_previsores].copy()
classe = base[col_classe].copy()


# =============================================================================
#      Transformar as variáveis categóricas (ordinais) em valores numéricos
# =============================================================================

# Ordem natural de severidade para as variáveis ordinais
ordem_stress = ["Low", "Medium", "High"]
ordem_addiction = ["None", "Mild", "Moderate", "Severe"]

ordinal_encoder = OrdinalEncoder(categories=[ordem_stress, ordem_addiction])
previsores[["stress_level", "addiction_level"]] = ordinal_encoder.fit_transform(
	previsores[["stress_level", "addiction_level"]]
)

previsores["stress_level"] = previsores["stress_level"].astype("int64")
previsores["addiction_level"] = previsores["addiction_level"].astype("int64")


# =============================================================================
#      Transformar as variáveis categóricas (nominais) em variáveis numéricas
# =============================================================================

labelencoder_gender = LabelEncoder()
previsores["gender"] = labelencoder_gender.fit_transform(previsores["gender"])
previsores["gender"] = previsores["gender"].astype("int64")

# academic_work_impact tem semântica binária (Yes/No), então LabelEncoder é suficiente
labelencoder_impact = LabelEncoder()
previsores["academic_work_impact"] = labelencoder_impact.fit_transform(
	previsores["academic_work_impact"]
)
previsores["academic_work_impact"] = previsores["academic_work_impact"].astype("int64")


# =============================================================================
#                     Balanceamento com Undersampling
# =============================================================================

risk_count = classe.value_counts()
print("Classe 0:", risk_count.get(0, 0))
print("Classe 1:", risk_count.get(1, 0))

undersample = RandomUnderSampler(random_state=0)
previsores_balanceado, classe_balanceada = undersample.fit_resample(previsores, classe)

print("\nDistribuição após undersampling:")
print(classe_balanceada.value_counts())


# =============================================================================
#                     Exportando base balanceada
# =============================================================================

df_balanceado = previsores_balanceado.copy()
df_balanceado[col_classe] = classe_balanceada
df_balanceado.to_csv("dataset_preprocessado_balanceado.csv", index=False)


# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
	previsores_balanceado,
	classe_balanceada,
	test_size=0.25,
	random_state=0,
	stratify=classe_balanceada,
)


# =============================================================================
#                     Padronização dos dados
# =============================================================================

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)

print("\nShapes finais:")
print("Treinamento previsores:", previsores_treinamento.shape)
print("Teste previsores:", previsores_teste.shape)
print("Treinamento classe:", classe_treinamento.shape)
print("Teste classe:", classe_teste.shape)

