import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

arquivo = pd.read_csv('wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

# Separando as variáveis entre preditoras e variável alvo

y = arquivo['style']
x = arquivo.drop('style', axis=1)

# Criando os conjuntois de dados de treino e teste:

x_treino, x_teste, y_treino, y_teste, = train_test_split(x, y, test_size=0.3)

# Criação do modelo:
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)
# imprimindo resultados:
resultado = modelo.score(x_teste, y_teste)
print('Acurácia:', resultado)


print(y_teste[400:403], x_teste[400:403])

precisoes = modelo.predict(x_teste[400:403])
print(precisoes)

