
from efficientGP.data.load_design_module import load_design_csv

target_file = 'data/Pump_data/design_variable.csv'
design_keys, design_variable = load_design_csv(target_file)
print(design_keys)
print(design_variable[0])


