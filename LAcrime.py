import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
pd.set_option('display.max_columns', 500) # чтобы не было ...  при отображении таблиц
pd.set_option('display.width', 1000) # чтобы не было ...  при отображении таблиц
LAcrime = pd.read_csv("la-crimes-sample.csv", sep=",")
print(LAcrime.shape) #(39608, 27)
print(LAcrime.dtypes) #типы данных различных столбцов
print(LAcrime.head())
print(LAcrime.info()) # сколько уникальных значений в каждом столбце
print(LAcrime.isnull().sum()) #Сколько пропущенных
#Анализ распределения числовых переменных
#LAcrime["Time Occurred"].plot(kind="kde", xlim=(LAcrime["Time Occurred"].min(), LAcrime["Time Occurred"].max()))

LAcrime["Victim Age"].plot(kind="kde", xlim=(LAcrime["Victim Age"].min(), LAcrime["Victim Age"].max()))
LAcrime["Victim Sex"].value_counts().plot(kind="pie", figsize=(7, 7), fontsize=20)
#plt.show()
print(LAcrime["Victim Sex"].value_counts())
LAcrime2 = LAcrime[(LAcrime["Victim Sex"] == "M") | (LAcrime["Victim Sex"] == "F")]
LAcrime2["Victim Sex"].value_counts().plot(kind="pie", figsize=(7, 7), fontsize=20)
plt.show()
#LAcrime["Time Occurred"].hist()
LAcrime["Victim Age"].hist()
plt.show()
print(LAcrime["Victim Descent"].value_counts()) #Люди какого происхождения чаще всего являются жертвами преступлений?
print(LAcrime["Victim Sex"].value_counts()) #Верно ли, что женщины чаще оказываются жертвами по сравнению с мужчинами?
LAcrime[LAcrime["Victim Sex"] == "M"]["Victim Age"].hist()#В каком возрастном промежутке мужчины чаще становятся жетрвами преступлений?
plt.show()
print(LAcrime[LAcrime["Victim Sex"] == "M"]["Victim Age"].describe())
print(LAcrime.sample(30))
print(LAcrime["Crime Code Description"].value_counts()) #Определите 10 самых распространённых преступлений в LA. Постройте график.
#LAcrime["Crime Code Description"].value_counts().plot(kind="bar", figsize=(7, 7), fontsize=5) #все столбики
#LAcrime["Crime Code Description"].value_counts().loc[lambda x: x>1000].plot(kind="bar", figsize=(7, 7), fontsize=20) #только столбики, частота (value_counts) у которых >1
LAcrime["Crime Code Description"].value_counts().loc[lambda x: x>1000][:10].plot(kind="bar", figsize=(10, 10), fontsize=7, position = 1)
plt.show()

print(LAcrime[LAcrime["Victim Sex"] == "M"]["Crime Code Description"].value_counts()[:3])
print(LAcrime[LAcrime["Victim Sex"] == "F"]["Crime Code Description"].value_counts()[:3])
#От каких преступлений чаще старадют женщины, а от каких мужчины?
print('lalal')
print(LAcrime.groupby("Victim Sex")["Crime Code Description"].value_counts().loc[lambda x: x>1200])

#Отсортируйте районы, по количество преступлений. Постройте график, показывающий самые безопасные и опасные районы.
LAcrime.groupby("Area Name").agg({"DR Number": "count"})
LAcrime3=LAcrime.groupby("Area Name").agg({"DR Number": "count"}).sort_values(by="DR Number", ascending=False)
LAcrime3["Number of crimes"]=pd.concat([LAcrime3[0:3],LAcrime3[18:]])
print('koalkl')

LAcrime3.drop("DR Number", axis=1,inplace=True) # inplace=True - чтобы из исходной таблицы сохранились изменения в той же таблице
#print(LAcrime3["Number of crimes"])
LAcrime3.dropna(inplace=True)
print(LAcrime3)
LAcrime3.plot(kind="bar")
plt.show()
LAcrime3["Area risk"]= pd.Series(["HR", "HR", "HR", "LR", "LR", "LR"], index=LAcrime3.index) #новый столбик
print("vf")
print(LAcrime3)

#LAcrime3.dropna().drop("DR Number", axis=1).plot(kind='bar')

#plt.xlabel('Number of requests every 10 minutes')
#plt.legend(h1+h2, l1+l2, loc=2)


#Люди какого происхождения чаще всего страдают от преступлений в каждом из районов? Не забудьте нормировать на общее количество жертв.
print("lklk")
print(LAcrime.groupby("Area Name")["Victim Descent"].value_counts(normalize=True)) #нормировать на количество value_counts по  каждой группе по которой сгруппировано
# for group_name, group in LAcrime.groupby("Area Name")["Victim Descent"].value_counts():
#     print(group_name, group)
print(LAcrime.groupby("Area Name").agg({"Victim Descent": "count"}))
print("olololololo")
print(LAcrime.groupby(["Area Name"]).apply(lambda x: x["Victim Descent"].value_counts(normalize=True)[0])) # [0] - first element (the highest frequency) in value_counts
print(LAcrime.groupby(["Area Name"]).apply(lambda x: x["Victim Descent"].value_counts(normalize=True).index[0])) #происхождение людей, которые
#чаще всего страдают от преступлений в каждом из районов
