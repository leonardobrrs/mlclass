import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

print("Lendo os ficheiros originais...")
df_train = pd.read_csv('diabetes_dataset.csv')
df_app = pd.read_csv('diabetes_app.csv')

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# 1. TRATAR ZEROS FALSOS: Substituir por 'NaN' (Vazio)
print("Removendo falsos zeros biológicos...")
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_train[cols_with_zeros] = df_train[cols_with_zeros].replace(0, np.nan)
df_app[cols_with_zeros] = df_app[cols_with_zeros].replace(0, np.nan)

X_train = df_train[features]
y_train = df_train['Outcome']
X_app = df_app[features]

# 2. IMPUTAÇÃO ROBUSTA: Usar a Mediana 
print("Preenchendo lacunas com a Mediana...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_app_imputed = imputer.transform(X_app)

# 3. NORMALIZAÇÃO: StandardScaler
print("Normalizando a escala (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_app_scaled = scaler.transform(X_app_imputed)

# 4. RECONSTRUIR
print("Guardando os dados...")
df_train_final = pd.DataFrame(X_train_scaled, columns=features)
df_train_final['Outcome'] = y_train.values

df_app_final = pd.DataFrame(X_app_scaled, columns=features)

df_train_final.to_csv('diabetes_dataset.csv', index=False)
df_app_final.to_csv('diabetes_app.csv', index=False)

print("Pronto! Execute o diabetes_csv.py.")