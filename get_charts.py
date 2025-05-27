from bestdori.charts import Chart
from bestdori.songs import *

songs = get_all()

for i in songs.keys():
    for j in songs[i]["difficulty"].keys():
        chart = Chart.get_chart(int(i), ['easy', 'normal', 'hard', 'expert', 'special'][int(j)])
        if (chart is not None) and (not chart.is_sp_rhythm):
            chart_json = chart.json()
            with open(f'charts/{i}-{j}_{songs[i]["difficulty"][j]["playLevel"]}.json', 'w', encoding='utf-8') as f:
                f.write(chart_json)
                print(f'Chart id {i} difficulty {j} saved.')