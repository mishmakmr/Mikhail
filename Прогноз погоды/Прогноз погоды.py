"""
Инструкция
Напишите программу на языке Python, которая выводит прогноз температуры в вашем городе на сегодня в формате
T +/- d
Где T - средняя температура в этот день за последние 9 лет, d - среднеквадратичное отклонение (по ГОСТу) температуры за последние 9 лет.
Для работы программы вам потребуется собрать данные по температуре в вашем городе в текущем месяце за последние 9 лет и организовать их в CSV или TSV формате следующего вида:
1;T1_1;T1_2;...;T1_9
2;T2_1;T2_2;...;T2_9
..
31;T31_1;T31_2;...;T31_9
Где T1_1 - температура в первый день месяца 10 лет назад, T1_2 - температура в первый день месяца 9 лет назад и т.д. T31_9 - температура в 31 день месяца год назад.
Если в текущем месяце меньше 31 дня, то в файле с данными будет меньше строк - по числу дней в текущем месяце.
В поле Отчет скопируйте ваше программу, в качестве приложения загрузите CSV/TSV файл с данными.
"""
import pandas as pd
from datetime import datetime
f = pd.read_csv("weather_2014_2023.csv", na_values="NA", delimiter=";",
                   decimal=",", skiprows = 6) # рабочий вариант
# print(f.head(50))
data = pd.DataFrame(f)
# print("1 вывод",data)
data ["T"] = data["T"].astype(float)
# print(data.dtypes)
# print(data["T"])
data = data[['Местное время в Архангельске / им. Ф. А. Абрамова (аэропорт)', 'T']]
data.rename(columns={'Местное время в Архангельске / им. Ф. А. Абрамова (аэропорт)': 'Date'}, inplace=True)
# print("2 вывод")
# print(data)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data['Day'] = pd.to_datetime(data['Date']).dt.day
data['Month'] = pd.to_datetime(data['Date']).dt.month
data['Year'] = pd.to_datetime(data['Date']).dt.year
# print("3 вывод")
# print(data)
# data1 =round(data.groupby(["Year", "Month", "Day"]).mean(), 2)
data1 = round(data.groupby(["Day", "Month", "Year"]).mean("T"), 2)
# print("4 вывод")
# print(data1)
# print(data1.dtypes)
#_____________________________________
# # Группируем по дню и месяцу и рассчитываем среднюю температуру и стандартное отклонение
daily_temp_stats = data1.groupby(['Month', 'Day']).mean('T')
# print(daily_temp_stats)
daily_temp_stats = round(data1.assign(std=data1['T'].std()).reset_index(), 2)
# print(daily_temp_stats)
#___________________
# Получаем сегодняшнюю дату и день месяца
today = datetime.now()
today_day = today.day
# print(today_day)
today_month = today.month
# print(today_month)
# _______________________________проверка, указание даты в ручную
# today = datetime.now()
# today_day = 31
# print(today_day)
# today_month = 12
# print(today_month)
#________________________________
# Получаем среднюю температуру и стандартное отклонение для сегодняшней даты
mean_temp = daily_temp_stats[(daily_temp_stats["Month"] == today_month) & (daily_temp_stats["Day"] == today_day)]["T"].values[0]
std_dev_temp = daily_temp_stats[(daily_temp_stats['Month'] == today_month) &
                                 (daily_temp_stats['Day'] == today_day)]['std'].values[0]
# Форматируем вывод
forecast = f"{mean_temp:.1f} +/- {std_dev_temp:.1f}"

print(f"Прогноз температуры на сегодня: {forecast}")
# ________________________________ сохраняем в файл
daily_temp_stats = daily_temp_stats.set_index("Day")
df = pd.DataFrame(daily_temp_stats)
df.to_csv(r'res.csv', sep=';', index=True)  # запись в файл
# print(df)
