import numpy as np


def load_DOE_csv(path):
    with open(path) as f:
        lines = [line for line in f.readlines() if line and not '#' in line]
    keys = [t.strip() for t in lines[0].strip().split(',')]
    values = [[float(v) for v in line.strip().split(',')] for line in lines[1:]]
    return keys, np.array(values, dtype=np.float32)



if __name__ == "__main__":
    csv_path = 'data/Pump_data/DOE_data.csv'
    metadata_keys, metadata_values = load_DOE_csv(csv_path)
    print(metadata_keys)
    print(metadata_values[0])
