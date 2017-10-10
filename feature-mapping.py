import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import pickle

ORIG_NAMES = [
	'a01',
	'bl_bi24latitude',
	'bl_bi24longitude',
	'bl_dw15',
	'bl_dw16',
	'bl_dw19',
	'bl_dw20',
	'bl_dw21',
	'bl_dw23',
	'bl_dw25',
	'bl_dw31',
	'bl_dw32',
	'bl_dw33',
	'bl_dw34',
	'bl_dw36',
	'bl_dw39',
	'bl_dw40',
	'bl_dw41',
	'bl_dw42',
	'bl_dw44',
	'bl_dw45',
	'bl_dw46',
	'bl_dw47',
	'bl_dw52',
	'bl_dw53',
	'bl_dw54',
	'bl_dw56',
	'bl_dw57',
	'bl_dw59',
	'bl_dw61',
	'bl_dw63',
	'bl_dw64',
	'bl_dw72',
	'bl_dw73',
	'bl_sd13',
	'bl_sd14',
	'bl_sd15',
	'bl_sd24_piazza',
	'bl_sd24_leghar',
	'bl_sd26',
	'bl_sd27',
	'bl_sd28',
	'bl_sd33',
	'bl_sd35',
	'bl_sd36',
	'bl_sd43',
	'bl_sd46',
	'bl_sd48',
	'bl_sd49',
	'bl_sd50',
	'bl_sd51'
]
NEW_NAMES = [
	'ID',
	'lat',
	'long',
	'water_source_drinking',
	'water_location_drinking',
	'water_interruptions_drinking',
	'water_unavailable',
	'water_treated',
	'water_quality_concerns',
	'water_satisfaction',
	'water_drinking_cooking_same',
	'water_source_cooking',
	'water_location_cooking',
	'water_distance_cooking',
	'water_interruptions_cooking',
	'toilet_type',
	'pit_latrine_depth',
	'toilet_shared',
	'toilet_num_households',
	'garbage_disposal',
	'garbage_distance',
	'garbage_regular',
	'garbage_frequency',
	'waste_water_disposal',
	'storm_drain',
	'other_drain',
	'electricity_access',
	'electricity_source',
	'electricity_payment',
	'electricity_always',
	'electricity_interruptions',
	'electricity_outages_length',
	'lighting_source',
	'cooking_fuel',
	'road_type',
	'road_safety',
	'road_rainy_season',
	'distance_piazza',
	'distance_leghar',
	'distance_minibus',
	'distance_motorcycle',
	'distance_light_rail',
	'healthcare_used',
	'healthcare_distance',
	'healthcare_satisfaction',
	'education_used',
	'education_distance',
	'satisfaction_num_classrooms',
	'satisfaction_num_textbooks',
	'satisfaction_cost_textbooks',
	'satisfaction_education'
]
OTHER = -999
DONT_KNOW = -888
NA = -777
TOO_MANY = -333

def main():
	data_labeled = pd.read_csv('Addis_data_withlabel.csv')
	data_labeled = data_labeled.fillna("empty")
	data = pd.read_csv('Addis_data.csv')
	data = data.fillna(-1)
	freq_list = []

	for i, col in enumerate(ORIG_NAMES):
		col_data = data_labeled[col]
		mode = stats.mode(col_data)
		freqs = Counter(col_data)
		freq_list.append((NEW_NAMES[i],freqs))

	pickle.dump(freq_list, open('freqs.pkl', 'wb'))


if __name__ == "__main__":
	main()
