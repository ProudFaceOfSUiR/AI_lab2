import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())  # just for understanding, what is happening
print(test_data.head())


def missing_value_checker(data):
    list = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            sum = data[feature].isna().sum()

            type = data[feature].dtype

            print(f'{feature}: {sum}, type: {type}')

            list.append(feature)
    print(list)

    print(len(list))


missing_value_checker(test_data)

test_edited = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)  # removing the titles
train_edited = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)


def nan_filler(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.median())
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes + 1


nan_filler(test_edited)
nan_filler(train_edited)

missing_value_checker(test_edited)

missing_value_checker(train_edited)

# train_edited.shape, test_edited.shape

print("Here comes the info")

test_edited.info()

train_edited.info()

X = train_edited.drop('SalePrice', axis=1)
y = train_edited['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

X_train.shape
test_edited.shape

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential(Dense(32, input_dim=75), )
#model.add(layers.Dense(64, activation='relu'))
# Добавим другой слой:
#model.add(layers.Dense(64, activation='relu'))
# Добавим слой softmax с 10 выходами:
#model.add(layers.Dense(10, activation='softmax'))
# замените None на колличество входных полносвязных слоёв, колличество нейронов, колличество выходов
#tf.random.set_seed(40) #/ torch.manual_seed(40) #Для обеспечения воспроизводимости результатов устанавливается функция seed

model.compile(optimizer="rmsprop",  loss="MSLE", metrics=["mae"])
model.summary()
# Для оценки потерь рекомендую использовать MSLE(MeanSquaredLogarithmicError), а также метрику MAE(Mean absolute error).

history = model.fit(X_train, y_train, epochs=500, batch_size=32)  # замените None на гиперпараметры вашей модели нейронной сети

print("plot next")
pd.DataFrame(history.history).plot()
plt.ylabel('accuracy')
plt.xlabel('epoch')
print(history.history)

#mport matplotlib.pyplot as plt
plt.show()

scores = model.evaluate(X_val, y_val, verbose=1)

preds = model.predict(test_edited)


print()

print("here comes preds")

print(preds)

print(pd.DataFrame)
output = pd.DataFrame(
    {
        'Id': test_data['Id'],
    })
print(output)
output = pd.DataFrame(np.squeeze(preds))
print(output)

pd.DataFrame(np.squeeze(preds)).plot()
plt.ylabel('shit1')
plt.xlabel('shit2')
#mport matplotlib.pyplot as plt
plt.show()
# output
# print (output)
