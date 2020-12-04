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

ser1 = pd.Series([1, 2, 3, 4], index=['USA', 'Germany', 'USSR', 'Japan'])
print(ser1)
'''
USA 1
Germany 2
USSR 3
Japan 4
dtype: int64'''

ser2 = pd.Series([1, 2, 5, 4], index=['USA', 'Germany', 'Italy', 'Japan'])
'''
USA 1
Germany 2
Italy 5
Japan 4
dtype: int64'''
print(ser2)
ser1['USA'] #1

ser1 + ser2
'''
Germany 4.0
Italy NaN
Japan 8.0
USA 2.0
USSR NaN
dtype: float64'''


print("\n"*2)
titanic_full_df = pd.read_csv("https://nagornyy.me/datasets/titanic.csv", sep=",") #разделить dataframe по запятым
print(titanic_full_df)
#quick view on data:
titanic_full_df.shape #(891, 12)
titanic_full_df.info() #инфа про каждый столбец
titanic_full_df.dtypes # тип данных в каждом из столбцов

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




###Индексация и выделение:
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
titanic_full_df.iloc[[564, 442]]# покажет 564ую и 442ую строчку со всеми колонками
titanic_full_df.loc[[564, 442], ["Name", "Sex"]]  # покажет 564ую и 442ую строчку, колонки ["Name", "Sex"]
titanic_full_df == 1 # выдаст таблицу из false и true (там где значение ==1)
titanic_full_df[titanic_full_df["Survived"] == 0].head() #Пассажиры, кто не выжил
titanic_full_df[titanic_full_df["Survived"] == 1]["Sex"].value_counts() #посчитать число значений
'''
female 233
male 109
Name: Sex, dtype: int64'''

'''nunique:
Syntax: Series.nunique(dropna=True)
Parameters:
dropna: Exclude NULL value if True
Return Type: Integer – Number of unique values in a column.'''

titanic_full_df[(titanic_full_df["Fare"] > 100)
                | (titanic_full_df["Name"].str.find("Master") != -1)].head()  # таблица где либо fare>100,
# либо в имени пассажира есть слово Master





###Методы:
titanic_full_df["Embarked"].unique()
'''array(['S', 'C', 'Q', nan], dtype=object)'''
titanic_full_df["Embarked"].nunique() #3


titanic_full_df["Survived"].value_counts()
'''
0 549
1 342
Name: Survived, dtype: int64'''


titanic_full_df["Pclass"].value_counts()
'''3 491
1 216
2 184
Name: Pclass, dtype: int64'''

titanic_full_df["Pclass"].replace({1: "Элита", 2: "Средний класс", 3: "Работяги"}, inplace=True)
#inplace, всегда по умолчанию False, что означает, что исходный DataFrame нетронутый,
# и операция возвращает новый DF. При настройке inplace = True операция может работать на исходном DF

titanic_full_df["Pclass"].value_counts()
'''Работяги 491
Элита 216
Средний класс 184
Name: Pclass, dtype: int64'''

titanic_full_df["Fare"].apply(lambda x: "Дёшево" if x < 20 else "Дорого")
'''
0 Дёшево
1 Дорого
2 Дёшево
3 Дорого
4 Дёшево
5 Дёшево
6 Дорого...
'''
titanic_full_df["Fare_Bin"] = titanic_full_df["Fare"].apply(lambda x: "Дёшево" if x < 20 else "Дорого")
#записать предыдущее в отдельный столбик
titanic_full_df.sort_values(by="Fare", ascending=False) #сортировка по значению колонки



###Работа с пропущенными значениями
titanic_full_df.isnull().any() #
#Отсутствующие данные помечаются как NaN.
# isnull() который сохраняет значение True для любого значения NaN
#Получаем таблицу того же размера, но на месте реальных данных в ней находятся логические переменные,которые
# принимают значение False, если значение поля у объекта есть,или True - если значение в поле NaN.

''' True - для столбцов где есть пропущенные значения
...
Pclass False
Name False
Sex False
Age True
SibSp False...
'''


titanic_full_df.dropna().head() #dropna - удаление строк с как минимум одним NAN.
#axis{0 or ‘index’, 1 or ‘columns’}, default 0
#dropna(axis = 0) - axis = 0 or ‘index’ : Drop rows which contain missing values.
# axis = 1, or ‘columns’ : Drop columns which contain missing value.

#how{‘any’, ‘all’}, default ‘any’
# how='all' - удаляет строки/колонки, где все элементы NAN
# ‘any’ : какой-то из элементов NAN
#subset - Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include.
titanic_full_df.dropna(subset=["Age", "Sex"]).head() #удаляем только те строчки, у которых есть NAN в столбцах ["Age", "Sex"]
titanic_full_df.dropna(thresh=12).head() # не менее 12 заполненных колонок (без NAN)  #Keep only the rows with at least 12 non-NA values/
titanic_full_df.fillna("ПРОПУСК").head() #Заменить все NAN на слово "ПРОПУСК" #Fill NA/NaN values using the specified method
titanic_full_df["Age"].mean() #среднее значение по столбцу
titanic_full_df["Age"].fillna(value=titanic_full_df["Age"].mean()).head() #заменить в столбце Age все NAN а среднее значение по столбцу Age
titanic_full_df[["Sex", "Survived"]].pivot_table(index=["Sex"], columns=["Survived"], aggfunc=len)
#pivot table - сводная таблица,
'''
Survived	0	1
             Sex		
  female	81	233
  male	   468	109'''


titanic_full_df[["Sex", "Survived", "Age"]].pivot_table(values=["Age"], index=["Sex"],
                                                        columns=["Survived"], aggfunc="mean")


'''	
Age
Survived	0	1
               Sex		
female	25.046875	28.847716
male	31.618056	27.276022'''


titanic_full_df.groupby("Pclass")
#A groupby operation involves some combination of splitting the object, applying a function,
#and combining the results. This can be used to group large amounts of data and compute operations
#on these groups.

print(titanic_full_df.groupby("Pclass").mean()["Age"])
'''
Pclass
Работяги 25.140620
Средний класс 29.877630
Элита 38.233441
Name: Age, dtype: float64'''

titanic_full_df[["Pclass", "Age"]].pivot_table(values=["Age"], index=["Pclass"], aggfunc="mean") #тот же рез-ат, что
#в предыд команде
'''              Age
Pclass                  
Работяги       25.140620
Средний класс  29.877630
Элита          38.233441'''


titanic_full_df.groupby("Pclass").mean().loc["Работяги"] # только по одному конкретномук значению из Pclass
'''PassengerId 439.154786
Survived 0.242363
Age 25.140620
SibSp 0.615071
Parch 0.393075
Fare 13.675550
Relatives 1.008147
Name: Работяги, dtype: float64'''

###Другие функции: count, min/max, describe(), first, std ...

titanic_full_df.groupby("Pclass").describe()["Age"] #описательная статистика по выбранным колонкам (по которым группируем) и признакам
'''	
count	mean	std	min	25%	50%	75%	max
Pclass								
Работяги	355.0	25.140620	12.495398	0.42	18.0	24.0	32.0	74.0
Средний класс	173.0	29.877630	14.001077	0.67	23.0	29.0	36.0	70.0
Элита	186.0	38.233441	14.802856	0.92	27.0	37.0	49.0	80.0
'''

titanic_full_df.groupby("Pclass").describe()["Age"].transpose()
'''
Pclass	Работяги	Средний класс	Элита
count	355.000000	173.000000	186.000000
mean	25.140620	29.877630	38.233441
std	12.495398	14.001077	14.802856
min	0.420000	0.670000	0.920000
25%	18.000000	23.000000	27.000000
50%	24.000000	29.000000	37.000000
75%	32.000000	36.000000	49.000000
max	74.000000	70.000000	80.000000'''

titanic_full_df.groupby("Pclass")["Age"].agg(["min", "max", "std"])
'''
               min  	max	   std
Pclass			
Работяги	   0.42	   74.0	  12.495398
Средний класс	0.67	70.0	14.001077
Элита	        0.92	80.0	14.802856'''

titanic_full_df.groupby("Pclass").agg({"Age": np.mean, "PassengerId": "count"}) #.agg - агрегирование, np.mean - функция агрегирования
#"count" - Count non-NA cells for each column or row.
'''	
             Age	PassengerId
Pclass		
Работяги	   25.140620	491
Средний класс	29.877630	184
Элита	        38.233441	216'''



titanic_full_df.groupby(["Pclass", "Sex"]).mean()["Fare"]
'''Pclass Sex
Работяги female 16.118810
male 12.661633
Средний класс female 21.970121
male 19.741782
Элита female 106.125798
male 67.226127
Name: Fare, dtype: float64'''




###Цикл по значениям
###Если вы исползуете циклы, возможно вы что-то делаете не так. Иногда, однако, это необходимо.

print(ser1)
'''
USA 1
Germany 2
USSR 3
Japan 4
dtype: int64'''
for (index, value) in ser1.iteritems():
    print("Страна {}, место {}.".format(index, value))  # iteritems() - Iterate over (column name, Series) pairs.
'''
Страна USA, место 1.
Страна Germany, место 2.
Страна USSR, место 3.
Страна Japan, место 4.'''


for index, row in titanic_full_df.iterrows():   #iterrows() - Iterate over DataFrame rows as (index, Series) pairs.
    print(index, row["Name"])
'''
0 Braund, Mr. Owen Harris
1 Cumings, Mrs. John Bradley (Florence Briggs Thayer)
2 Heikkinen, Miss. Laina
3 Futrelle, Mrs. Jacques Heath (Lily May Peel)
4 Allen, Mr. William Henry
5 Moran, Mr. James
6 McCarthy, Mr. Timothy J
7 Palsson, Master. Gosta Leonard
8 Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
9 Nasser, Mrs. Nicholas (Adele Achem)
10 Sandstrom, Miss. Marguerite Rut
...'''


for group_name, group in titanic_full_df.groupby("Pclass"):
    print(group_name, group["Age"].mean())
'''
Работяги 25.14061971830986
Средний класс 29.87763005780347
Элита 38.233440860215055'''


for group_name, group in titanic_full_df.groupby("Pclass"):
    print(group_name, group["Age"].mean())
'''
Работяги 25.14061971830986
Средний класс 29.87763005780347
Элита 38.233440860215055'''


###Слияние и соединение