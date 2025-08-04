
import pyvista as pv
import numpy as np




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

print('Done')