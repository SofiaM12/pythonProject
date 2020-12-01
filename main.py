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
print(titanic_full_df.columns) # Index (['PassengerId', 'Survived', 'Pclass' ... 'Embarked'], dtype='object')