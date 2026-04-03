from torch.utils.data import ConcatDataset
from utils import *

def load_fmri_train(timeshift, delay, batch_size, train_data, train_num, thred=1):
    timeshift = timeshift + 1
    path = './dataset/' + train_data + '.npz'
    file_struct = np.load(path)
    x_struct = file_struct['struct_data']

    x_struct = x_struct[200:1000, :train_num, :]
    x_struct = x_struct.astype(np.float32)
    data_wd = []
    for i in range(x_struct.shape[0]-delay+1):
        temp = []
        for j in range(delay):
            temp.append(x_struct[i+j, :, :])
        temp = np.stack(temp, 0)
        data_wd.append(temp)
    data_wd = np.stack(data_wd, 0)
    data_wd = np.transpose(data_wd, (0, 2, 1, 3))

    data = []
    for i in range(x_struct.shape[0] - timeshift - delay + 2):
        temp = []
        for j in range(timeshift):
            temp.append(data_wd[i+j, :, :, :])
        data.append(temp)

    data_struct = np.stack(data, 0)
    data_struct = np.transpose(data_struct, (0, 2, 1, 4, 3))  # data shape:[time_series, subjects, train_time, dims, embed_time]
    data_struct = data_struct.reshape(-1, *data_struct.shape[2:])
    data_struct = torch.from_numpy(data_struct.astype(np.float32))
    train_loader_struct = torch.utils.data.DataLoader(data_struct, batch_size=batch_size, drop_last=True, shuffle=True,
                                                      num_workers=0)

    adj_tuple, edge_value, adj = load_dti(thred)
    adj_tuple = torch.tensor(adj_tuple)
    adj_tuple = adj_tuple.long()
    edge_value = torch.tensor(edge_value)
    edge_value = edge_value.float()
    return train_loader_struct, adj_tuple, edge_value


def load_fmri_test(start_time, predict_time, delay, batch_size, task, test_num=-1, thred=1, surrogate='no'):
    path = './dataset/' + task + '.npz'
    file_struct = np.load(path)
    x_struct = file_struct['fmri_data']
    x_struct = x_struct.astype(np.float32, copy=False)

    if surrogate == 'phase':
        x_struct = phase_random_surrogate(x_struct)
    elif surrogate == 'time':
        x_struct = x_struct.copy()
        for s in range(x_struct.shape[1]):
            perm = np.random.permutation(x_struct.shape[0])
            x_struct[:, s, :] = x_struct[perm, s, :]

    x_test = x_struct[start_time:start_time+predict_time+delay, :test_num, :]
    x_test = x_test.astype(np.float32)
    dataset_struct = []
    for i in range(x_test.shape[0]-delay+1):
        temp = []
        for j in range(delay):
            temp.append(x_test[i+j, :, :])
        temp = np.stack(temp, 0)
        dataset_struct.append(temp)

    dataset_struct = np.stack(dataset_struct, 0)
    dataset_struct = np.transpose(dataset_struct, (2, 0, 3, 1))
    dataset_struct = torch.tensor(dataset_struct).float()
    loader_struct = torch.utils.data.DataLoader(dataset_struct, batch_size=10, shuffle=False, drop_last=False, num_workers=0)

    adj_tuple, edge_value, adj = load_dti(thred)
    adj_tuple = torch.tensor(adj_tuple)
    adj_tuple = adj_tuple.long()
    edge_value = torch.tensor(edge_value)
    edge_value = edge_value.float()
    return loader_struct, adj_tuple, edge_value, adj

def load_fmri_time(predict_time, delay, awake_delay, batch_size, task, subtask, test_num, timerow, need_compare, thred=0):
    path = 'D:/github/gcn_np/dataset/' + task + '.npz'
    file_struct = np.load(path)
    subject = file_struct['id'][:test_num]
    x_test = file_struct['struct_data'][:, :test_num, :]
    if subtask != 'no':
        x_test, ids = select_subject_time(x_test, subject, predict_time, delay, awake_delay, task, subtask, timerow, need_compare)

    dataset_struct = []
    for i in range(x_test.shape[0] - delay + 1):
        temp = []
        for j in range(delay):
            temp.append(x_test[i + j, :, :])
        temp = np.stack(temp, 0)
        dataset_struct.append(temp)

    dataset_struct = np.stack(dataset_struct, 0)
    dataset_struct = np.transpose(dataset_struct, (2, 0, 3, 1))
    dataset_struct = torch.tensor(dataset_struct).float()
    combined = BrainDataset(dataset_struct, subject)
    loader_struct = torch.utils.data.DataLoader(combined, batch_size=batch_size, shuffle=False, drop_last=False,
                                                num_workers=0)


    adj_tuple, edge_value, adj = load_dti(thred)
    adj_tuple = torch.tensor(adj_tuple)
    adj_tuple = adj_tuple.long()
    edge_value = torch.tensor(edge_value)
    edge_value = edge_value.float()
    return loader_struct, adj_tuple, edge_value, adj, ids


if __name__ == '__main__':
    pass
