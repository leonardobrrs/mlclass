import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

print("Lendo os arquivos originais...")
df_train = pd.read_csv('diabetes_dataset.csv')
df_app = pd.read_csv('diabetes_app.csv')

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 1. TRATAR OS ZEROS FALSOS: 
# Nestas colunas, '0' significa 'Não Medido' (NaN)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_train[cols_with_zeros] = df_train[cols_with_zeros].replace(0, np.nan)
df_app[cols_with_zeros] = df_app[cols_with_zeros].replace(0, np.nan)

X_train = df_train[features]
y_train = df_train['Outcome']
X_app = df_app[features]

# 2. IMPUTAÇÃO INTELIGENTE (KNNImputer)
print("Fazendo imputação inteligente (KNNImputer)...")
imputer = KNNImputer(n_neighbors=5)
X_train_imputed = imputer.fit_transform(X_train)
X_app_imputed = imputer.transform(X_app)

# 3. NORMALIZAÇÃO MIN-MAX
print("Aplicando Min-Max Scaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_app_scaled = scaler.transform(X_app_imputed)

# 4. CÁLCULO AUTOMÁTICO DE PESOS VIA CORRELAÇÃO
print("\nAnalisando a importância de cada variável (Correlação)...")
# Criamos um DataFrame temporário com os dados já preenchidos para a matemática ficar perfeita
df_temp = pd.DataFrame(X_train_imputed, columns=features)
df_temp['Outcome'] = y_train.values

# Calcula a correlação (valor absoluto) com o Outcome e remove a própria coluna Outcome da lista
correlacoes = df_temp.corr()['Outcome'].abs().drop('Outcome')
print(correlacoes)

# Converte os valores da correlação num array e multiplica pela base normalizada
pesos = correlacoes.values
X_train_scaled = X_train_scaled * pesos
X_app_scaled = X_app_scaled * pesos

# 5. RECONSTRUIR E SALVAR OS FICHEIROS
print("\nReconstruindo e salvando ficheiros CSV...")
df_train_final = pd.DataFrame(X_train_scaled, columns=features)
df_train_final['Outcome'] = y_train.values

df_app_final = pd.DataFrame(X_app_scaled, columns=features)

df_train_final.to_csv('diabetes_dataset.csv', index=False)
df_app_final.to_csv('diabetes_app.csv', index=False)

print("Tudo pronto! Ficheiros CSV altamente otimizados. Pode rodar o código da atividade.")