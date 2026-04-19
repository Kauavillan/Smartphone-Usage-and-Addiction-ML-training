import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler


# =============================================================================
#                    Configuração de colunas opcionais
# =============================================================================

considerar_daily_screen_time_hours = True
considerar_addiction_level = False


# =============================================================================
#                         Leitura da base e resumo
# =============================================================================
base = pd.read_csv("datasets/raw_dataset.csv")
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
	"social_media_hours",
	"gaming_hours",
	"work_study_hours",
	"sleep_hours",
	"notifications_per_day",
	"app_opens_per_day",
	"weekend_screen_time",
	"stress_level",
	"academic_work_impact",
]

if considerar_daily_screen_time_hours:
	cols_previsores.append("daily_screen_time_hours")

if considerar_addiction_level:
	cols_previsores.append("addiction_level")

col_classe = "addicted_label"

previsores = base[cols_previsores].copy()
classe = base[col_classe].copy()


# =============================================================================
#      Transformar as variáveis categóricas (ordinais) em valores numéricos
# =============================================================================

# Ordem natural de severidade para as variáveis ordinais
ordem_stress = ["Low", "Medium", "High"]
ordem_addiction = ["None", "Mild", "Moderate", "Severe"]

if considerar_addiction_level:
	ordinal_encoder = OrdinalEncoder(categories=[ordem_stress, ordem_addiction])
	previsores[["stress_level", "addiction_level"]] = ordinal_encoder.fit_transform(
		previsores[["stress_level", "addiction_level"]]
	)
else:
	ordinal_encoder = OrdinalEncoder(categories=[ordem_stress])
	previsores[["stress_level"]] = ordinal_encoder.fit_transform(
		previsores[["stress_level"]]
	)

previsores["stress_level"] = previsores["stress_level"].astype("int64")
if considerar_addiction_level:
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

undersample = RandomUnderSampler(random_state=0)
previsores_balanceado, classe_balanceada = undersample.fit_resample(previsores, classe)


# =============================================================================
#                     Exportando base balanceada
# =============================================================================

df_balanceado = previsores_balanceado.copy()
df_balanceado[col_classe] = classe_balanceada
df_balanceado.to_csv("datasets/dataset_preprocessado_balanceado.csv", index=False)


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


