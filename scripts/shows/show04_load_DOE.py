


from efficientGP.data.load_DOE_module import load_DOE_csv

csv_path = 'data/Pump_data/DOE_data.csv'
metadata_keys, metadata_values = load_DOE_csv(csv_path)
print(metadata_keys)
print(metadata_values[0])