
import numpy as np



def load_cfdpost(path):
    with open(path) as file:
        lines = file.readlines()
    
    data_dict = {}
    i = 0
    mode = None
    name = None
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line[0] == '[':
            if line.startswith('[Name]'):
                name = lines[i + 1].strip()
                data_dict[name] = {}
                i += 1
            elif line.startswith('[Data]'):
                mode = 'data'
                data_dict[name]['key'] = [k.strip() for k in lines[i + 1].strip().split(',')]
                data_dict[name]['value'] = []
                i += 1
            elif line.startswith('[Faces]'):
                mode = 'face'
                data_dict[name]['face'] = []
            i += 1
            continue

        if mode == 'data':
            data_dict[name]['value'].append([float(v) for v in line.split(',')])
        elif mode == 'face':
            data_dict[name]['face'].append([int(idx) for idx in line.split(',')])
        else:
            print(f'line[{i + 1}] Unknown format!')
            print(line)
            break
        i += 1

    for name in data_dict:
        data_dict[name]['value'] = np.asarray(data_dict[name]['value'], dtype='float32')
        data_dict[name]['face'] = np.asarray(data_dict[name]['face'], dtype='int32')
    return data_dict



if __name__ == "__main__":
    data_path = 'data/toy/data/impeller/impeller_DP0.csv'
    data = load_cfdpost(data_path)
    print(data['impeller']['key'])
    print(data['impeller']['value'][0])
    print(data['impeller']['face'][0])