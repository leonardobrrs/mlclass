import pandas as pd
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

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
print("=" * 70)
print("COMPARANDO MÚLTIPLOS ALGORITMOS COM 10-FOLD CROSS VALIDATION")
print("=" * 70)
print()

# Dicionário para armazenar resultados
resultados = {}

# ============================================================
# 3.1 RANDOM FOREST (sem normalização)
# ============================================================
print("-" * 70)
print("1. RANDOM FOREST")
print("-" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring="accuracy")

resultados['Random Forest'] = {
    'scores': rf_scores,
    'model': rf_model,
    'scaler': None,
    'X': X
}

print(f"Acurácias por fold: {[round(s, 4) for s in rf_scores]}")
print(f"Acurácia média:     {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"Melhor fold:        {rf_scores.max():.4f}")
print(f"Pior fold:          {rf_scores.min():.4f}")
print()

# ============================================================
# 3.2 SUPPORT VECTOR MACHINE - SVM (com StandardScaler)
# ============================================================
print("-" * 70)
print("2. SUPPORT VECTOR MACHINE (SVM) - Com StandardScaler")
print("-" * 70)

# Normalizar dados para SVM
scaler_svm = StandardScaler()
X_scaled_svm = scaler_svm.fit_transform(X)

svm_model = SVC(
    kernel='rbf',
    C=100,
    gamma='scale',
    random_state=42,
    verbose=0
)

svm_scores = cross_val_score(svm_model, X_scaled_svm, y, cv=skf, scoring="accuracy")

resultados['SVM'] = {
    'scores': svm_scores,
    'model': svm_model,
    'scaler': scaler_svm,
    'X': X_scaled_svm
}

print(f"Acurácias por fold: {[round(s, 4) for s in svm_scores]}")
print(f"Acurácia média:     {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
print(f"Melhor fold:        {svm_scores.max():.4f}")
print(f"Pior fold:          {svm_scores.min():.4f}")
print()

# ============================================================
# 3.3 REDE NEURAL ARTIFICIAL - MLPClassifier (com StandardScaler)
# ============================================================
print("-" * 70)
print("3. REDE NEURAL ARTIFICIAL (MLPClassifier) - Com StandardScaler")
print("-" * 70)

# Normalizar dados para Rede Neural
scaler_mlp = StandardScaler()
X_scaled_mlp = scaler_mlp.fit_transform(X)

mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    verbose=0
)

mlp_scores = cross_val_score(mlp_model, X_scaled_mlp, y, cv=skf, scoring="accuracy")

resultados['MLPClassifier'] = {
    'scores': mlp_scores,
    'model': mlp_model,
    'scaler': scaler_mlp,
    'X': X_scaled_mlp
}

print(f"Acurácias por fold: {[round(s, 4) for s in mlp_scores]}")
print(f"Acurácia média:     {mlp_scores.mean():.4f} ± {mlp_scores.std():.4f}")
print(f"Melhor fold:        {mlp_scores.max():.4f}")
print(f"Pior fold:          {mlp_scores.min():.4f}")
print()

# ============================================================
# 3.4 RESUMO COMPARATIVO
# ============================================================
print("=" * 70)
print("RESUMO COMPARATIVO")
print("=" * 70)

comparacao = []
for nome, dados in resultados.items():
    comparacao.append({
        'Algoritmo': nome,
        'Acurácia Média': f"{dados['scores'].mean():.4f}",
        'Desvio Padrão': f"{dados['scores'].std():.4f}",
        'Melhor': f"{dados['scores'].max():.4f}",
        'Pior': f"{dados['scores'].min():.4f}"
    })

df_comparacao = pd.DataFrame(comparacao)
print(df_comparacao.to_string(index=False))
print()

# Encontrar melhor modelo
melhor_algoritmo = max(resultados.items(), key=lambda x: x[1]['scores'].mean())
print(f"🏆 Melhor algoritmo: {melhor_algoritmo[0]} (Acurácia: {melhor_algoritmo[1]['scores'].mean():.4f})")
print()

# ============================================================
# 4. TREINAR O MELHOR MODELO NO DATASET COMPLETO
# ============================================================
print("=" * 70)
print(f"TREINANDO {melhor_algoritmo[0].upper()} NO DATASET COMPLETO...")
print("=" * 70)

melhor_nome = melhor_algoritmo[0]
melhor_dados = melhor_algoritmo[1]
melhor_modelo = melhor_dados['model']
melhor_scaler = melhor_dados['scaler']

# Se o melhor modelo requer escalonamento, fazer fit no scaler com dados completos
if melhor_scaler is not None:
    X_train_final = melhor_scaler.fit_transform(X)
else:
    X_train_final = X

# Treinar o modelo final
melhor_modelo.fit(X_train_final, y)

# Verificar importância das features (se disponível)
if hasattr(melhor_modelo, 'feature_importances_'):
    importances = pd.Series(melhor_modelo.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print()
    print("Importância das features:")
    for feat, imp in importances.items():
        barra = "█" * int(imp * 50)
        print(f"  {feat:<20} {imp:.4f}  {barra}")
elif hasattr(melhor_modelo, 'coef_'):
    # Para modelos como SVM e MLP que usam coeficientes
    print()
    print("Importância das features (baseado em coeficientes):")
    coef_abs = np.abs(melhor_modelo.coef_[0] if len(melhor_modelo.coef_.shape) > 1 else melhor_modelo.coef_)
    importances = pd.Series(coef_abs, index=X.columns)
    importances = importances.sort_values(ascending=False)
    for feat, imp in importances.items():
        barra = "█" * int((imp / coef_abs.max()) * 50)
        print(f"  {feat:<20} {imp:.4f}  {barra}")
print()

# ============================================================
# 5. GERAR PREVISÕES PARA O APP
# ============================================================
print("=" * 70)
print("GERANDO PREVISÕES COM O MODELO SELECIONADO...")
print("=" * 70)

# Se o modelo precisa de normalização, aplicar o scaler aos dados de teste
if melhor_scaler is not None:
    app_processed = melhor_scaler.transform(app)
else:
    app_processed = app

# Fazer previsões
predictions = melhor_modelo.predict(app_processed)

print(f"Modelo utilizado:    {melhor_nome}")
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

print("=" * 70)
print("CONCLUÍDO!")
print(f"Acurácia estimada (melhor modelo): {melhor_algoritmo[1]['scores'].mean():.2%}")
print(f"Modelo selecionado: {melhor_nome}")
print("=" * 70)