import pyvista as pv
import numpy as np



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


if __name__ == "__main__":
    
    from efficientGP.data.load_cfdpost_module import load_cfdpost

    data_path = 'data/toy/data/impeller/impeller_DP0.csv'
    
    data = load_cfdpost(data_path)
    multiblock = data_to_multiblock(data)

    # multiblock.plot(cpos='xy', window_size=[300, 300], jupyter_backend='static')
    multiblock.plot(window_size=[500, 500], jupyter_backend='static', show_edges=True, zoom=3, scalars='Wall Shear Z [ Pa ]', cmap='jet')