import pandas as pd

data = pd.read_csv('ufc_fight_stat_data.csv')

data.drop('fight_stat_id', inplace=True, axis=1)

print(data)

data.to_csv('modified_ufc_fight_stat_data.csv', index=False)