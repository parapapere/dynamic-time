from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

s = np.array([0, 2, 0, 1, 0, 0], dtype=np.double)
t = np.array([0, 0, 0.5, 2, 0, 1, 0], dtype=np.double)

path = dtw.warping_path(s, t)
cost = dtw.distance(s,t)

print(f"Cost: \n\n{cost} \n\nPath: \n\n{path}")

dtwvis.plot_warping(s, t, path, filename="warp.png")