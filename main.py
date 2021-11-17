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
    '''
    validation of the data
    :param data:
    :return:
    '''
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
    '''
    also a filter
    :param data:
    :return:
    '''
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


from keras.models import Sequential
from keras.layers import Dense



model = Sequential(Dense(100, input_dim=75))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='softplus'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(1))

'''
Все параметры подобраны эксперементально, в т. ч. функции активации
'''

model.compile(optimizer="rmsprop",  loss="MSLE", metrics=["mae"])
model.summary()


history = model.fit(X_train, y_train, epochs=10, batch_size=10)

print("plot next")
pd.DataFrame(history.history).plot()
plt.ylabel('mid squared errors')
plt.xlabel('epoch')
print(history.history)


plt.show()

scores = model.evaluate(X_val, y_val, verbose=1)

preds = model.predict(test_edited)


print()

print("here comes preds")

print(preds)


output = pd.DataFrame(
{
    'Id':test_data['Id'],
    'SalePrice': np.squeeze(preds)
})
output
print(output)

'''
Ниже идет вариант, чтобы строить график относительно цены, позволяет визуально сравнить результаты
Особого практического смысла нет, т к нет ярко выраженной зависимости
'''

'''
print(pd.DataFrame)
output = pd.DataFrame(
{
    'Id':test_data['Id'],
    'SalePrice': np.squeeze(preds)
})
output
output = pd.DataFrame(np.squeeze(preds))
print(output)

pd.DataFrame(np.squeeze(preds)).plot()
plt.ylabel('Y')
plt.xlabel('X')
#mport matplotlib.pyplot as plt
plt.show()
#output
# print (output)
'''
