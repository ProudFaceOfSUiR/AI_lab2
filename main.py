import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tf as tf
from keras import layers
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.head()
test_data.head()


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

test_edited = test_data.drop(['Alley','FireplaceQu','PoolQC', 'Fence', 'MiscFeature'], axis=1)
train_edited = train_data.drop(['Alley','FireplaceQu','PoolQC', 'Fence', 'MiscFeature'], axis=1)

def nan_filler(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.median())
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes+1

nan_filler(test_edited)
nan_filler(train_edited)

missing_value_checker(test_edited)

missing_value_checker(train_edited)

train_edited.shape, test_edited.shape

test_edited.info()

train_edited.info()

X = train_edited.drop('SalePrice', axis=1)
y = train_edited['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

X_train.shape, test_edited.shape

from tensorflow import keras# или import torch
import torch
import tensorflow as tf
model = keras.Sequential(layers.Dense(64, activation='relu'))
# замените None на колличество входных полносвязных слоёв, колличество нейронов, колличество выходов
#tf.random.set_seed(40) / torch.manual_seed(40)
#Для обеспечения воспроизводимости результатов устанавливается функция seed

model.compile(loss=['msle'], optimizer=tf.keras.optimizers.Adam(0.01), metrics=['mae'])
#Для оценки потерь рекомендую использовать MSLE(MeanSquaredLogarithmicError), а также метрику MAE(Mean absolute error).

history = model.fit(X_train, y_train, None) #замените None на гиперпараметры вашей модели нейронной сети

pd.DataFrame(history.history).plot()
plt.ylabel('accuracy')
plt.xlabel('epoch')
print(history.history)

scores = model.evaluate(X_val, y_val, verbose=1)

preds = model.predict(test_edited)
print(pd.DataFrame)
output = pd.DataFrame(
{
    'Id':test_data['Id'],
})
print(output)
output = pd.DataFrame(
{
    'SalePrice': np.squeeze(preds)
})
print(output)
#output
#print (output)