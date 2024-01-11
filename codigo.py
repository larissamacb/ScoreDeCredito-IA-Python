import pandas as pd

tabela = pd.read_csv("clientes.csv")
print(tabela)

# Tratamento dos dados

print(tabela.info())

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

for coluna in tabela.columns:
    if tabela[coluna].dtype == "object" and coluna != "score_credito":
        tabela[coluna] = codificador.fit_transform(tabela[coluna])

print(tabela.info())

x = tabela.drop(["score_credito", "id_cliente"], axis=1)
y = tabela["score_credito"]

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvore = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

from sklearn.metrics import accuracy_score

previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))

# Quais as características mais importantes para definir o score de crédito?

colunas = list(x_teste.columns)
importancia = pd.DataFrame(index=colunas, data=modelo_arvore.feature_importances_)
importancia = importancia * 100
print(importancia)

# Dívida total, mix de crédito e juros de empréstiimo

# Previsão dos novos clientes

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")

tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])
print(tabela_novos_clientes)

previsoes = modelo_arvore.predict(tabela_novos_clientes)
print(previsoes)