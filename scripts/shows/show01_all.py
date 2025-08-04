import torch
import pyvista as pv
import numpy as np
from pathlib import Path




points = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
faces = [3, 0, 1, 2]
mesh = pv.PolyData(points, faces)
mesh.plot(cpos='xy', window_size=[300, 300], jupyter_backend='static')


points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
# faces = [4, 0, 1, 2, 3]
faces = [3, 0, 1, 2, 3, 0, 2, 3]
mesh = pv.PolyData(points, faces)
mesh.triangulate(inplace=True)
mesh.plot(cpos='xy', window_size=[300, 300], jupyter_backend='static', style='wireframe')

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


data_path = 'data/toy/data/impeller/impeller_DP0.csv'
data = load_cfdpost(data_path)
print(data['impeller']['key'])
print(data['impeller']['value'][0])
print(data['impeller']['face'][0])

def data_to_multiblock(data):
    multiblock = pv.MultiBlock()
    for name, block_data in data.items():
        pos = block_data['value'][:, 1:4]
        face = np.pad(block_data['face'], [[0, 0], [1, 0]], constant_values=4).reshape(-1)
        block = pv.PolyData(pos, face)
        for i in range(4, len(block_data['key'])):
            block.point_data[block_data['key'][i]] = block_data['value'][:, i]

        point_area = np.zeros_like(block_data['value'][:, 1])
        face = np.array(block.faces).reshape(-1, 5)[:, 1:]
        face_area = block.compute_cell_sizes().cell_data['Area']
        face_normal = block.compute_normals(consistent_normals=False).cell_data['Normals']
        point_normal = np.zeros([block.n_points, 3], dtype=face_normal.dtype)
        point_area_sum = np.zeros([block.n_points], dtype=face_area.dtype)

        for f, n, a in zip(face, face_normal, face_area):
            point_area[f] += a

            point_normal[f] += a * n
            point_area_sum[f] += a

        point_area /= 4
        block.point_data['Area'] = point_area

        point_normal /= point_area_sum[:, None]
        point_normal /= np.linalg.norm(point_normal, ord=2, axis=-1, keepdims=True)
        block.point_data['Normals'] = point_normal

        multiblock.append(block, name=name)
    return multiblock

multiblock = data_to_multiblock(data)
# multiblock.plot(window_size=[500, 500], jupyter_backend='static', show_edges=True, zoom=3, scalars='Wall Shear Z [ Pa ]', cmap='jet')


def load_csv(path):
    with open(path) as f:
        lines = [line for line in f.readlines() if line and not '#' in line]
    keys = [t.strip() for t in lines[0].strip().split(',')]
    values = [[float(v) for v in line.strip().split(',')] for line in lines[1:]]
    return keys, np.array(values, dtype=np.float32)


# csv_path = 'data/DOE_data.csv'
# metadata_keys, metadata_values = load_csv(csv_path)
# print(metadata_keys)
# print(metadata_values[0])


def mean_std(arr, axis):
    return arr.mean(axis=axis), arr.std(axis=axis)

def pack(padded, num):
    return torch.cat([t[:n] for t, n in zip(padded, num)], dim=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_train = 8
num_val = 2
n_padded = 65000  # padding is needed (each mesh has different size)

metadata_keys, metadata_values = load_csv('data/DOE_data.csv')

multiblocks = []
for data_id in range(num_train + num_val):
    Path('data/pv/').mkdir(parents=True, exist_ok=True)
    multiblock = data_to_multiblock(load_cfdpost(f'data/impeller/impeller_DP{data_id}.csv'))
    multiblocks.append(multiblock)

xs, ys, ns = [], [], []
for multiblock in multiblocks:
    x = np.concatenate([multiblock[0].points, multiblock[0].point_data['Normals']], axis=-1)  # each point input feature = [position, normal] (3 + 3 dim)
    y = np.stack([multiblock[0].point_data[k] for k in ['Pressure [ Pa ]', 'Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear Z [ Pa ]']], axis=-1)  # each point output feature (4 dim)
    n = multiblock[0].n_points  # num of points
    xs.append(x)
    ys.append(y)
    ns.append(n)

x_mean, x_std = mean_std(np.concatenate(xs[:num_train], axis=0), axis=0)
x_mean[3:] = 0  # mean of normal feature = 0
x_std[3:] = 1  # stddev of normal feature = 1
y_mean, y_std = mean_std(np.concatenate(ys[:num_train], axis=0), axis=0)

x = np.stack([np.pad((x - x_mean) / x_std, [[0, n_padded - len(x)], [0, 0]]) for x in xs])
y = np.stack([np.pad((y - y_mean) / y_std, [[0, n_padded - len(y)], [0, 0]]) for y in ys])
ns = np.stack(ns).astype(np.int32)
x_train, x_val = x[:num_train], x[num_train:]
y_train, y_val = y[:num_train], y[num_train:]
n_train, n_val = ns[:num_train], ns[num_train:]

cs = np.array(metadata_values[:num_train + num_val, metadata_keys.index('RPM')], dtype=np.float32)[:, None]
c_mean, c_std = mean_std(cs[:num_train], axis=0)
c_train = (cs[:num_train] - c_mean) / c_std
c_val = (cs[num_train:] - c_mean) / c_std

ios = np.stack([metadata_values[:num_train + num_val, metadata_keys.index(k)] for k in ['Pt_in [Pa]', 'Pt_out [Pa]']], axis=-1).astype(np.float32)
io_mean, io_std = mean_std(ios[:num_train], axis=0)
io_train = (ios[:num_train] - io_mean) / io_std
io_val = (ios[num_train:] - io_mean) / io_std

[x_train, x_val, y_train, y_val, n_train, n_val, c_train, c_val, io_train, io_val, x_mean, x_std, y_mean, y_std, c_mean, c_std, io_mean, io_std] = [
    torch.tensor(t, device=device, dtype=torch.float32) for t in [x_train, x_val, y_train, y_val, n_train, n_val, c_train, c_val, io_train, io_val, x_mean, x_std, y_mean, y_std, c_mean, c_std, io_mean, io_std]]



class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_k = torch.nn.LayerNorm(dim)
        self.att = torch.nn.MultiheadAttention(dim, 1, batch_first=True)
        self.norm_ffn = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_ffn),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_ffn, dim),
        )

    def forward(self, q, k, key_padding_mask=None):
        qn = self.norm_q(q)
        kn = self.norm_k(k)
        x = q + self.att(qn, kn, kn, need_weights=False, key_padding_mask=key_padding_mask)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x
    
class PumpModel(torch.nn.Module):
    def __init__(self, n_query, n_self_attention, dim, n_head=1, dropout=0.0):
        super().__init__()
        dim_ffn = dim * 4
        self.embed_point = torch.nn.Linear(6, dim)
        self.encoder = CrossAttention(dim, dim_ffn)
        self.embed_condition = torch.nn.Linear(1, dim)
        self.pos_enc = PositionalEncoding(dim, max_len=n_query+3)
        self.self_attention = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(dim, n_head, dim_ffn, dropout, batch_first=True, norm_first=True), n_self_attention, enable_nested_tensor=False,
        )
        self.decoder = CrossAttention(dim, dim_ffn)

        self.linear_output = torch.nn.Linear(dim, 4)
        self.query = torch.nn.Parameter(torch.randn([1, n_query, dim]))

        self.inoutlet_token = torch.nn.Parameter(torch.randn([1, 2, dim]))
        self.norm_out = torch.nn.LayerNorm(dim)
        self.linear_inlet = torch.nn.Linear(dim, 1)
        self.linear_outlet = torch.nn.Linear(dim, 1)
    
    def forward(self, x, n, c):
        # [Input]
        # x shape: BatchSize X NumPoint(padded) X 6(pos + normal)
        # n shape: BatchSize; example = [2, 10, 5]
        # c shape: BatchSize X 1
        # [Output]
        # y shape: BatchSize X NumPoint(padded) X 4
        # io shape: BatchSize X 2
        q = self.query.repeat([x.shape[0], 1, 1])
        p = self.embed_point(x)
        padding_mask = torch.arange(x.shape[1]).to(x.device)[None] >= n[:, None]
        x = self.encoder(q, p, key_padding_mask=padding_mask)
        c = self.embed_condition(c)
        inoutlet_token = self.inoutlet_token.repeat([x.shape[0], 1, 1])
        x = torch.cat([c[:, None], inoutlet_token, x], dim=1)
        x = self.self_attention(self.pos_enc(x))
        x = self.norm_out(x)
        y = self.linear_output(self.decoder(p, x[:, 3:]))
        io = torch.cat([self.linear_inlet(x[:, 1]), self.linear_outlet(x[:, 2])], dim=-1)
        return y, io



model = PumpModel(n_query=64, n_self_attention=2, dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 1000 + 1):
    model.train()
    optimizer.zero_grad()
    idx = torch.randperm(len(x_train))
    y_pred, io_pred = model(x_train[idx], n_train[idx], c_train[idx])
    loss_y = torch.nn.functional.mse_loss(pack(y_pred, n_train[idx]), pack(y_train[idx], n_train[idx]))
    loss_io = torch.nn.functional.mse_loss(io_pred, io_train[idx])
    loss = loss_y + loss_io
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_pred, io_pred = model(x_val, n_val, c_val)
    y_mae = torch.abs((pack(y_pred, n_val)*y_std + y_mean) - (pack(y_val, n_val)*y_std + y_mean)).mean(dim=0)
    io_mae = torch.abs((io_pred*io_std + io_mean) - (io_val*io_std + io_mean)).mean(dim=0)

    print(f'epoch {epoch}] loss={loss.item():.4e}, MAE(y)={y_mae.numpy(force=True)}, MAE(io)={io_mae.numpy(force=True)}')

print('done')