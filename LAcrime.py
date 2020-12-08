import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
pd.set_option('display.max_columns', 500) # чтобы не было ...  при отображении таблиц
pd.set_option('display.width', 1000) # чтобы не было ...  при отображении таблиц
LAcrime = pd.read_csv("la-crimes-sample.csv", sep=",")
print(LAcrime.shape) #(39608, 27)
LAcrime.info()
print(LAcrime.dtypes) #типы данных различных столбцов
print(LAcrime.head())