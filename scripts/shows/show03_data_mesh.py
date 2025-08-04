

from efficientGP.data.load_cfdpost_module import load_cfdpost
from efficientGP.data.data_to_multiblock_module import data_to_multiblock

data_path = 'data/toy/data/impeller/impeller_DP0.csv'
data = load_cfdpost(data_path)
print(data['impeller']['key'])
print(data['impeller']['value'][0])
print(data['impeller']['face'][0])





data_path = 'data/toy/data/impeller/impeller_DP0.csv'

data = load_cfdpost(data_path)
multiblock = data_to_multiblock(data)

# multiblock.plot(cpos='xy', window_size=[300, 300], jupyter_backend='static')
multiblock.plot(window_size=[500, 500], jupyter_backend='static', show_edges=True, zoom=3, scalars='Wall Shear Z [ Pa ]', cmap='jet')