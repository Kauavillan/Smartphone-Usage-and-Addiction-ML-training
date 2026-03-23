import pandas as pd
base = pd.read_csv('credit_data.csv')
resumo = base.describe()

# =============================================================================
#                     Tratando valores inválidos
# =============================================================================

# Verifica idades inválidas
base.loc[base['Age'] < 0]

# Apagar a coluna
# base.drop('Age', axis=1, inplace=True)

# Apagar apenas os registros com problema
# base.drop(base[base['Age'] < 0].index, inplace=True)

# Preencher os valores com a média 
idade_media = base['Age'][base['Age'] > 0].mean().round()
base.loc[base['Age'] < 0, 'Age'] = idade_media

# Procurando as colunas que possuem algum valor faltante
pd.isnull(base).any()

# Preencher os valores com o mais frequente
saving_maioria = base['Saving accounts'][pd.notnull(base['Saving accounts'])].describe().top
base.loc[pd.isnull(base['Saving accounts']), 'Saving accounts'] = saving_maioria

checking_maioria = base['Checking account'][pd.notnull(base['Checking account'])].describe().top
base.loc[pd.isnull(base['Checking account']), 'Checking account'] = checking_maioria

# =============================================================================
#                     Separando dados em previsores e classe
# =============================================================================

cols_previsores = ['Age','Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose']
cols_classe = ['Risk']

previsores = base[cols_previsores].copy()
classe = base[cols_classe].copy()

# =============================================================================
#      Transformar as variáveis categóricas (ordinais) em valores numéricos
# =============================================================================

from sklearn.preprocessing import LabelEncoder
import numpy as np

# Instanciar LabelEncoder para cada variável categórica
labelencoder_sex = LabelEncoder()
labelencoder_housing = LabelEncoder()
labelencoder_purpose = LabelEncoder()
labelencoder_saving_accounts = LabelEncoder()
labelencoder_checking_account = LabelEncoder()
labelencoder_risk = LabelEncoder()

# Codificação da variável 'Sex' (assumindo 'male' e 'female') 
previsores.loc[:, 'Sex'] = labelencoder_sex.fit_transform(previsores.loc[:, 'Sex']) # female=0, male=1 (ordem alfabética)
previsores['Sex'] = previsores['Sex'].astype('int64')

# Codificação da variável 'Housing' (opcional, se houver)
# previsores.loc[:, 'Housing'] = labelencoder_housing.fit_transform(previsores.loc[:, 'Housing'])
# previsores['Housing'] = previsores['Housing'].astype('int64')

# Codificação ordinal para 'Saving accounts'
labelencoder_saving_accounts.classes_ = np.array(['little', 'moderate', 'rich', 'quite rich'])
previsores.loc[:, 'Saving accounts'] = labelencoder_saving_accounts.transform(previsores.loc[:, 'Saving accounts'])
previsores['Saving accounts'] = previsores['Saving accounts'].astype('int64')

# Codificação ordinal para 'Checking account'
labelencoder_checking_account.classes_ = np.array(['little', 'moderate', 'rich', 'quite rich'])
previsores.loc[:, 'Checking account'] = labelencoder_checking_account.transform(previsores.loc[:, 'Checking account'])
previsores['Checking account'] = previsores['Checking account'].astype('int64')

# Codificação da variável 'Purpose' (opcional, se houver)
# previsores.loc[:, 'Purpose'] = labelencoder_purpose.fit_transform(previsores.loc[:, 'Purpose']).astype('int64')
# previsores['Purpose'] = previsores['Purpose'].astype('int64')

# Codificação da variável alvo 'Risk' (assumindo 'good' e 'bad')
classe.loc[:, 'Risk'] = labelencoder_risk.fit_transform(classe.loc[:, 'Risk']) # bad=0, good=1 (ordem alfabética)
classe['Risk'] = classe['Risk'].astype('int64')

# =============================================================================
#      Transformar as variáveis categóricas (nominais) em variáveis dummy
# =============================================================================

from sklearn.preprocessing import LabelBinarizer
labelbinarizer = LabelBinarizer()

# Variavel Purpose
variaveis_dummy = labelbinarizer.fit_transform(previsores['Purpose'])
novas_variaveis_dummy = labelbinarizer.classes_
df_variaveis_dummy = pd.DataFrame(variaveis_dummy, columns=novas_variaveis_dummy)
previsores = previsores.join(df_variaveis_dummy)
previsores = previsores.drop('Purpose',axis=1)

# Variavel Housing
variaveis_dummy = labelbinarizer.fit_transform(previsores['Housing'])
novas_variaveis_dummy = labelbinarizer.classes_
df_variaveis_dummy = pd.DataFrame(variaveis_dummy, columns=novas_variaveis_dummy)
previsores = previsores.join(df_variaveis_dummy)
previsores = previsores.drop('Housing',axis=1)

cols_previsores = previsores.columns

# =============================================================================
#                     Balanceamento com Undersampling
# =============================================================================

risk_count = base['Risk'].value_counts()
print('Classe good:', risk_count['good'])
print('Classe bad:', risk_count['bad'])
print('Proportion:', round(risk_count['good'] / risk_count['bad'], 2), ': 1')
risk_count.plot(kind='bar', title='Count (target)',color = ['#1F77B4', '#FF7F0E']);

from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

# Aplicando Undersampling
undersample = RandomUnderSampler(random_state=0)
previsores, classe = undersample.fit_resample(previsores, classe)

# Exibindo as novas distribuições das classes
print(classe.value_counts())

# Exportar o DataFrame para um arquivo CSV
df.to_csv('credit_data_balanceada.csv', index=False)


# =============================================================================
#                 Separando em base de testes e treinamento
# =============================================================================

#  usando 25% para teste
from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


# =============================================================================
#                     Padronização dos dados
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores_treinamento = scaler.fit_transform(previsores_treinamento)
previsores_teste = scaler.transform(previsores_teste)






