import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ============================================================
# 1. CARREGAR OS DADOS
# ============================================================
print("=" * 60)
print("CARREGANDO DADOS...")
print("=" * 60)

df  = pd.read_csv("abalone_dataset.csv")
app = pd.read_csv("abalone_app.csv")

print(f"Dataset de treino:  {df.shape[0]} linhas, {df.shape[1]} colunas")
print(f"Dataset de app:     {app.shape[0]} linhas, {app.shape[1]} colunas")
print()
print("Distribuição das classes:")
print(df["type"].value_counts().sort_index())
print()

# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
print("=" * 60)
print("PRÉ-PROCESSAMENTO...")
print("=" * 60)

le = LabelEncoder()

# Codifica a coluna 'sex' (M, F, I → 0, 1, 2)
df["sex"]  = le.fit_transform(df["sex"])
app["sex"] = le.transform(app["sex"])

# Separar features e target
X = df.drop("type", axis=1)
y = df["type"]

print(f"Features: {list(X.columns)}")
print(f"Classes:  {list(y.unique())}")
print()

# ============================================================
# 3. VALIDAÇÃO COM STRATIFIED K-FOLD (10 folds)
# ============================================================
print("=" * 60)
print("VALIDANDO O MODELO (10-Fold Cross Validation)...")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

print(f"Acurácias por fold: {[round(s, 4) for s in scores]}")
print(f"Acurácia média:     {scores.mean():.4f}")
print(f"Desvio padrão:      {scores.std():.4f}")
print(f"Melhor fold:        {scores.max():.4f}")
print(f"Pior fold:          {scores.min():.4f}")
print()

# ============================================================
# 4. TREINAR NO DATASET COMPLETO
# ============================================================
print("=" * 60)
print("TREINANDO NO DATASET COMPLETO...")
print("=" * 60)

model.fit(X, y)

# Verificar importância das features
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

print("Importância das features:")
for feat, imp in importances.items():
    barra = "█" * int(imp * 50)
    print(f"  {feat:<20} {imp:.4f}  {barra}")
print()

# ============================================================
# 5. GERAR PREVISÕES PARA O APP
# ============================================================
print("=" * 60)
print("GERANDO PREVISÕES...")
print("=" * 60)

predictions = model.predict(app)

print(f"Total de previsões:  {len(predictions)}")
print(f"Distribuição:        {dict(zip(*np.unique(predictions, return_counts=True)))}")
print(f"Primeiras 10:        {predictions[:10].tolist()}")
print()

# ============================================================
# 6. SALVAR PREVISÕES EM CSV
# ============================================================
resultado = pd.DataFrame({"type": predictions})
resultado.to_csv("predictions.csv", index=False)
print("Previsões salvas em: predictions.csv")
print()

# ============================================================
# 7. ENVIAR PARA O SERVIDOR (descomente quando quiser enviar)
# ============================================================
# ATENÇÃO: só 1 envio a cada 12h! Verifique a acurácia antes.

def enviar_previsoes(predictions_array):
    """
    Envia as previsões para o servidor da atividade.
    Descomente a chamada no final para enviar.
    """
    URL    = "https://aydanomachado.com/mlclass/03_Validation.php"
    TEAM   = "_Null"   # ← coloque o nome do seu time aqui

    lista = [int(p) for p in predictions_array]

    payload = {
        "dev_key": TEAM,
        "predictions": json.dumps(lista)  # "[1,2,3,1...]"
    }

    try:
        response = requests.post(URL, data=payload)
        print("=" * 60)
        print("RESPOSTA DO SERVIDOR:")
        print("=" * 60)
        print(response.text)
    except Exception as e:
        print(f"Erro ao enviar: {e}")


# Descomente a linha abaixo para enviar as previsões:
enviar_previsoes(predictions)

print("=" * 60)
print("CONCLUÍDO!")
print(f"Acurácia estimada: {scores.mean():.2%}")
print("=" * 60)