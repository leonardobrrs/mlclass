import pandas as pd

# 1. Lê o arquivo
df = pd.read_csv('diabetes_dataset.csv')

# 2. Calcula a correlação de todas as colunas em relação ao 'Outcome' (o resultado)
# Pegamos os valores absolutos (abs) porque correlações negativas (inversas) também são importantes
correlacoes = df.corr()['Outcome'].abs().sort_values(ascending=False)

print("Importância de cada variável para prever a Diabetes:")
print(correlacoes)