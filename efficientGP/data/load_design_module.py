import numpy as np


def load_design_csv(path):
    with open(path) as f:
        # lines = [line for line in f.readlines() if line and not '#' in line]
        lines = f.readlines()
    keys = [t.strip() for t in lines[0].strip().split(',')]
    values = [[float(v) for v in line.strip().split(',')] for line in lines[1:]]
    return keys, np.array(values, dtype=np.float32)


if __name__ == "__main__":
    
    target_file = 'data/Pump_data/design_variable.csv'
    design_keys, design_variable = load_design_csv(target_file)
    print(design_keys)
    print(design_variable[0])