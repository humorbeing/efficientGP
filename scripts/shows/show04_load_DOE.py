


from efficientGP.data.load_DOE_module import load_DOE_csv


target_file = 'data/Pump_data/DOE_data.csv'
from efficientGP import AP
target_file = AP(target_file)


metadata_keys, metadata_values = load_DOE_csv(target_file)
print(metadata_keys)
print(metadata_values[0])