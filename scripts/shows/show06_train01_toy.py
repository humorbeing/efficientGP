import torch
import numpy as np
import glob


from efficientGP.data.load_DOE_module import load_DOE_csv

from efficientGP.data.load_cfdpost_module import load_cfdpost
from efficientGP.data.data_to_multiblock_module import data_to_multiblock

from efficientGP.transforms.mesh.utils import pack
from efficientGP.transforms.mesh.utils import pad_xs
from efficientGP.transforms.mesh.utils import retrieve_data


from efficientGP.models.model_v01 import PumpModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_train = 8
num_val = 2
n_padded = 65000  # padding is needed (each mesh has different size)
EPOCH = 5




DOE_csv_path = 'data/Pump_data/DOE_data.csv'

metadata_keys, metadata_values = load_DOE_csv(DOE_csv_path)


impeller_folder = 'data/toy/data/impeller'

file_list = glob.glob(impeller_folder + '/**/*.csv', recursive=True)
file_list.sort()




multiblocks = []
for f_ in file_list:    
    multiblock = data_to_multiblock(load_cfdpost(f_))
    multiblocks.append(multiblock)

xs, ys, ns = [], [], []
for multiblock in multiblocks:
    x = np.concatenate([multiblock[0].points, multiblock[0].point_data['Normals']], axis=-1)  # each point input feature = [position, normal] (3 + 3 dim)
    y = np.stack([multiblock[0].point_data[k] for k in ['Pressure [ Pa ]', 'Wall Shear X [ Pa ]', 'Wall Shear Y [ Pa ]', 'Wall Shear Z [ Pa ]']], axis=-1)  # each point output feature (4 dim)
    n = multiblock[0].n_points  # num of points
    xs.append(x)
    ys.append(y)
    ns.append(n)



x = pad_xs(xs)
y = pad_xs(ys)

ns = np.stack(ns).astype(np.int32)
x_train, x_val = x[:num_train], x[num_train:]
y_train, y_val = y[:num_train], y[num_train:]
n_train, n_val = ns[:num_train], ns[num_train:]



cs = retrieve_data(metadata_values, metadata_keys, 'RPM')[:, None]  # RPM as condition
cs = cs[:num_train + num_val]

c_train = cs[:num_train]
c_val = cs[num_train:]

io_in = retrieve_data(metadata_values, metadata_keys, 'Pt_in [Pa]')
io_in = io_in[:num_train + num_val]

io_out = retrieve_data(metadata_values, metadata_keys, 'Pt_out [Pa]')
io_out = io_out[:num_train + num_val]


ios = np.stack([io_in, io_out], axis=-1)

io_train = ios[:num_train] 
io_val = ios[num_train:]


[x_train, x_val, y_train, y_val, nt_train, nt_val, c_train, c_val, io_train, io_val] = [
    torch.tensor(t, device=device, dtype=torch.float32) for t in [x_train, x_val, y_train, y_val, n_train, n_val, c_train, c_val, io_train, io_val]]





model = PumpModel(n_query=64, n_self_attention=2, dim=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(1, EPOCH + 1):
    model.train()
    optimizer.zero_grad()
    idx = torch.randperm(len(x_train))
    x = x_train[idx]
    nt = nt_train[idx]
    n = n_train[idx]
    c = c_train[idx]
    ios = io_train[idx]

    y_pred, io_pred = model(x, nt, c)
    loss_y = torch.nn.functional.mse_loss(pack(y_pred, n), pack(y_train[idx], n))
    loss_io = torch.nn.functional.mse_loss(io_pred, ios)
    loss = loss_y + loss_io
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_pred, io_pred = model(x_val, nt_val, c_val)
    y_mae = torch.abs(pack(y_pred, n_val) - pack(y_val, n_val)).mean(dim=0)
    io_mae = torch.abs(io_pred - io_val).mean(dim=0)

    print(f'epoch {epoch}] loss={loss.item():.2e}, MAE(y)={y_mae.numpy(force=True)}, MAE(io)={io_mae.numpy(force=True)}')

print('done')