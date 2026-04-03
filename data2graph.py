import numpy as np

def adj_to_tuples(matrix):
    tuples = []
    value = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                value.append(matrix[i][j])
                tuples.append((i, j))
    return tuples, value

def load_dti(thred):
    print('Threshold of DTI: ', thred)
    path = './dataset/dti.npz'
    file = np.load(path)
    dti = file['dti_data']
    dti[dti < thred] = 0

    non_zero_count = np.count_nonzero(dti)
    zero_cols = np.sum(~dti.any(axis=0))

    print(f'Number of non-zero elements: {non_zero_count}')
    print(f'Number of full-zero columns: {zero_cols}')

    adj_tuple, value = adj_to_tuples(dti)
    adj_tuple = np.transpose(np.array(adj_tuple))
    edge_value = np.transpose(np.array(value))
    return adj_tuple, edge_value, dti

if __name__ == '__main__':
    pass


