import pandas as pd
import numpy as np

'''Series – это проиндексированный одномерный массив значений. Он похож на
простой словарь типа dict, где имя элемента будет соответствовать индексу,
а значение – значению записи.

DataFrame — это проиндексированный многомерный массив значений,
соответственно каждый столбец DataFrame, является структурой Series.'''

labels = ['a', 'b', 'c']
my_list = [10, 20, 30]
arr = np.array([10, 20, 30])
d = {'a': 10, 'b': 20, 'c': 30}
print(pd.Series(data=my_list))
'''
0    10
1    20
2    30
dtype: int64 '''
pd.Series(data=my_list, index=labels)

'''
a    10
b    20
c    30
dtype: int64'''
pd.Series(arr,labels)
'''
a 10
b 20
c 30
dtype: int64'''

pd.Series(d)
'''
a 10
b 20
c 30
dtype: int64'''


'''The Main Advantage of Pandas Series Over NumPy Arrays:
While we didn’t encounter it at the time, NumPy arrays are highly limited by one
characteristic: every element of a NumPy array must be the same type of data structure.
Said differently, NumPy array elements must be all string, or all integers, or all 
booleans. Pandas Series do not suffer from this limitation. 

As an example, you can pass three of Python’s built-in functions into a pandas 
Series without getting an error:'''
pd.Series([sum, print, len])
print("\n"*2)
titanic_full_df = pd.read_csv("https://nagornyy.me/datasets/titanic.csv", sep=",") #разделить dataframe по запятым
print(titanic_full_df)
#quick view on data:
titanic_full_df.shape #(891, 12)
titanic_full_df.info()

titanic_full_df.describe() # describe statistics on columns
titanic_full_df.columns # Index (['PassengerId', 'Survived', 'Pclass' ... 'Embarked'], dtype='object')

pd.set_option('display.max_columns', 500) # чтобы не было ...  при отображении таблиц
pd.set_option('display.width', 1000) # чтобы не было ...  при отображении таблиц

titanic_full_df.head() #верхняя часть таблицы (5 строк)
titanic_full_df.tail() #нижняя часть таблицы (5 строк)
titanic_full_df.isnull().sum() #Отсутствующие данные помечаются как NaN.
# isnull() который сохраняет значение True для любого значения NaN
#Получаем таблицу того же размера, но на месте реальных данных в ней находятся логические переменные,которые
# принимают значение False, если значение поля у объекта есть, или True, если значение в данном поле – это NaN.
# .sum - подсчет числа true столбцам (т.е. количество NAN в каждем столбце)


titanic_full_df.sample(5) #5 рандомных образцов

#Индексация и выделение:
titanic_full_df["Age"].head()
'''
0 22.0
1 38.0
2 26.0
3 35.0
4 35.0
Name: Age, dtype: float64'''

# Pass a list of column names:
titanic_full_df[["Age", "Sex"]].head()

#Добавить новый столбец:
titanic_full_df["Relatives"] = titanic_full_df["SibSp"] + titanic_full_df["Parch"]
print(titanic_full_df[["SibSp", "Parch", "Relatives"]].head())

titanic_full_df.drop("Relatives", axis=1).head()  #drop - удалить строки/столбцы,
# axis: int or string value, 0 ‘index’ for Rows and 1 ‘columns’ for Columns.

titanic_full_df.index.tolist()[:10] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
titanic_full_df.loc[442 : 450 : 2, ["Age", "Sex"]] # выбираем ряды по индексу
titanic_full_df.set_index(["Embarked"]).loc["S"].head() #вместо столбца с индексами (первого) - столбец embarked (порт) со значением только S
titanic_full_df.iloc[0] # для чтения и записи значения в датафрейм  /df.iloc[1, 1] = '21' задать значение в первой строке, первом столбце, нумерация строк и столбцов с нуля/
titanic_full_df.iloc[[564, 442]]# покажет 564ую и 442ую строчку
