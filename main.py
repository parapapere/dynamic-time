import numpy as np

def data_time_warping(x, y):
    n, m = len(x), len(y)

    #cost matrix
    cost_matrix = np.zeros((n+1, m+1))
    
    for i in range(1, n+1):
        cost_matrix[i, 0] = np.inf
    for j in range(1, m+1):
        cost_matrix[0, j] = np.inf
    
    #matrix for keeping traceback imformation
    traceback_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            
            dist = abs(x[i] - y[j])
            # print(dist, i, j, x[i], y[j])
            penalty = [
                cost_matrix[i, j],      # match (0)
                cost_matrix[i, j + 1],  # insertion (1)
                cost_matrix[i + 1, j]]  # deletion (2)
            
            min_penalty = np.argmin(penalty)
            cost_matrix[i + 1, j + 1] = dist + penalty[min_penalty]
            traceback_matrix[i, j] = min_penalty

    # traceback from bottom right
    i = n - 1
    j = m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_matrix[i, j]
        if tb_type == 0:
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            i = i - 1
        elif tb_type == 2:
            j = j - 1
        path.append((i, j))

    
    return cost_matrix[1:, 1:], path 

s = np.array([0, 2, 0, 1, 0, 0])
t = np.array([0, 0, 0.5, 2, 0, 1, 0])

cost, match = data_time_warping(s,t)

print(f"Cost matrix: \n\n{cost} \n\nPath: \n\n{match}")