import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

print("Lendo os arquivos originais...")
df_train = pd.read_csv('diabetes_dataset.csv')
df_app = pd.read_csv('diabetes_app.csv')

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 1. Separar as features (X) do alvo (y)
X_train = df_train[features]
y_train = df_train['Outcome']
X_app = df_app[features]

# 2. Imputação: Preencher os vazios com a Mediana (mais robusto que a média)
print("Preenchendo valores vazios...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
# Atenção: Usamos o transform() no app baseado no fit do train!
X_app_imputed = imputer.transform(X_app) 

# 3. Normalização: Colocar todas as variáveis na mesma escala (O SEGREDO DO K-NN!)
print("Normalizando a escala dos dados...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_app_scaled = scaler.transform(X_app_imputed)

# 4. Reconstruir os DataFrames com os dados super otimizados
df_train_final = pd.DataFrame(X_train_scaled, columns=features)
df_train_final['Outcome'] = y_train.values # Devolve a coluna Outcome intacta

df_app_final = pd.DataFrame(X_app_scaled, columns=features)

# 5. Sobrescrever os arquivos CSV para o código do professor ler
print("Salvando dados otimizados nos arquivos CSV...")
df_train_final.to_csv('diabetes_dataset.csv', index=False)
df_app_final.to_csv('diabetes_app.csv', index=False)

print("Tudo pronto! Pode rodar o script da atividade.")