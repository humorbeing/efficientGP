
from efficientGP.data.load_design_module import load_design_csv


target_file = 'data/Pump_data/design_variable.csv'
from efficientGP import AP
target_file = AP(target_file)

design_keys, design_variable = load_design_csv(target_file)
print(design_keys)
print(design_variable[0])


print('done')