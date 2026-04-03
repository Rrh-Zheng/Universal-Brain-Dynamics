from load import *
import time
from exp.exp_basic import Exp_Basic
from torch import optim
from scipy.io import savemat
from ubd_model import GraphConv
import os

class Exp_gcn(Exp_Basic):
    def __init__(self, args):
        super(Exp_gcn, self).__init__(args)

    def _build_model(self):
        model = GraphConv(self.args, self.args.enc_width, self.args.dec_width, self.args.aux_width)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def train(self):
        for num_idx in range(len(self.args.train_num_dict)):
            train_num = self.args.train_num_dict[num_idx]
            train_loader_struct, edge, value = load_fmri_train(self.args.time_shifts,
                                                                  self.args.delay, self.args.batch_size,
                                                                  self.args.train,
                                                                  train_num, self.args.thred)
            model_optim = self._select_optimizer()

            fold_path = './model/train_on_' + self.args.train + '_hidden_' + str(self.args.hidden_dim) + '/num_' + str(
                train_num) + '_thred_' + str(self.args.thred) + '/'
            print(fold_path)

            if not os.path.exists(fold_path):
                os.makedirs(fold_path)

            for epoch in range(self.args.epochs):
                self.model.train()
                t = time.time()
                epoch_loss_pred = 0
                epoch_loss_lin = 0

                for batch_idx, data in enumerate(train_loader_struct):
                    data, edge, value = data.to(self.device), edge.to(self.device), value.to(self.device)
                    dat, x_list, p_list, y_list = self.model(data, edge, value, self.args.time_shifts,
                                                             self.args.hidden_dim)

                    loss_pred = 0
                    loss_lin = 0

                    for i in range(0, self.args.time_shifts - 1):
                        loss_pred += F.mse_loss(torch.squeeze(x_list[i]), dat[i])

                    for i in range(self.args.time_shifts - 1):
                        loss_lin += F.mse_loss(y_list[i], p_list[i])

                    loss = loss_pred + loss_lin
                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()
                    epoch_loss_pred += loss_pred.item()
                    epoch_loss_lin += loss_lin.item()
                epoch_loss_pred = epoch_loss_pred / len(train_loader_struct)
                epoch_loss_lin = epoch_loss_lin / len(train_loader_struct)
                print("Epoch:", '%04d' % (epoch + 1), "train_loss_pred=", "{:.5f}".format(epoch_loss_pred),
                      "train_loss_lin=", "{:.5f}".format(epoch_loss_lin), "time=", "{:.5f}".format(time.time() - t))

                if (epoch + 1) % 100 == 0:
                    model_path = fold_path + str(epoch + 1) + '.pth'
                    torch.save(self.model.state_dict(), model_path)
                    print('save ' + str(epoch + 1) + ' successful')


    def predict(self):
        task = self.args.task_dict
        test_loader_struct, edge, value, _ = load_fmri_test( self.args.start_time,
                                                            self.args.predict_length,
                                                            self.args.delay, self.args.batch_size_test,
                                                            task, self.args.test_num, self.args.thred)


        model_path = './model/example.pth'
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        all_batch1 = []
        all_batch2 = []
        all_truths = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader_struct):
                truth = data[:, 1:self.args.predict_length+1, :, 0]
                all_truths.append(truth)
                data_struct, edge, value = data.to(self.device), edge.to(self.device), value.to(
                    self.device)
                x1_list, x2_list = self.model.predict(data_struct, edge, value, self.args.predict_length, self.args.hidden_dim)
                all_batch1.append(x1_list)
                all_batch2.append(x2_list)
        all_truths = torch.concatenate(all_truths).cpu()
        all_batch1 = torch.concatenate(all_batch1).cpu()
        all_batch2 = torch.concatenate(all_batch2).cpu()

        path = './predict/'
        if not os.path.exists(path):
            os.makedirs(path)

        path_full = path + task + '_subject_' + str(self.args.test_num) + '.mat'
        savemat(path_full, {'t': all_truths, 'x_adv': all_batch1, 'x_enc': all_batch2})

    def classfy(self):
        task = self.args.task_dict
        test_loader_struct, edge, value, _ = load_fmri_test(self.args.start_time,
                                                            self.args.predict_length,
                                                            self.args.delay, self.args.batch_size_test,
                                                            task, self.args.test_num, self.args.thred)
        torch.cuda.empty_cache()
        model_path = './model/example.pth'
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()


        x_shu_all = []
        x_seq_all = []
        e_shu_all = []
        e_seq_all = []

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader_struct):

                batch_data = data
                random_time = 100
                st = 20
                random_indices = torch.randperm(batch_data.shape[1])[:random_time]
                data_shu = batch_data[:, random_indices, :, :]
                data_seq = batch_data[:, st:st+random_time, :, :]
                data_shu, data_seq, edge, value = data_shu.to(self.device), data_seq.to(self.device), edge.to(
                    self.device), value.to(self.device)
                with torch.cuda.amp.autocast():
                    x_shu, x_seq, e_shu, e_seq = self.model.classfy(data_shu, data_seq, edge, value)

                x_shu_all.append(x_shu.cpu())
                x_seq_all.append(x_seq.cpu())
                e_shu_all.append(e_shu.cpu())
                e_seq_all.append(e_seq.cpu())

                del data_shu, data_seq, x_shu, x_seq, e_shu, e_seq
                torch.cuda.empty_cache()

            x_shu_all = torch.concatenate(x_shu_all).detach()
            x_seq_all = torch.concatenate(x_seq_all).detach()
            e_shu_all = torch.concatenate(e_shu_all).detach()
            e_seq_all = torch.concatenate(e_seq_all).detach()

            org_shu_name = 'x_shu_' + task
            enc_shu_name = 'e_shu_' + task
            org_seq_name = 'x_seq_' + task
            enc_seq_name = 'e_seq_' + task


            path = './classfy/subject_' + str(self.args.test_num) + '_length_' + str(random_time) + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            path_full = path + task + '_subject_' + str(self.args.test_num) + '.mat'
            savemat(path_full, {org_shu_name: x_shu_all, org_seq_name: x_seq_all, enc_shu_name: e_shu_all, enc_seq_name: e_seq_all})

    def cluster(self):
        for timerow_idx in range(len(self.args.timerow)):
            timerow = self.args.timerow[timerow_idx]
            model_path = './model/example.pth'
            print(model_path)
            self.model.load_state_dict(torch.load(model_path))
            for subtask_idx in range(len(self.args.subtask_dict)):
                task = self.args.task_dict
                subtask = self.args.subtask_dict[subtask_idx]
                test_loader_struct, edge, value, adj, ids = load_fmri_time(self.args.predict_length,
                                                                           self.args.delay, self.args.awake_delay,
                                                                           self.args.batch_size_test, task, subtask,
                                                                           self.args.test_num, timerow,
                                                                           False)
                x_list = []
                h_list = []
                for data, sub in test_loader_struct:
                    x_list.append(data[:, 1:, :, :])
                    data_struct, edge, value = data.to(self.device), edge.to(self.device), value.to(self.device)
                    h_adv = self.model.cluster(data_struct, edge, value, self.args.predict_length,
                                                         self.args.delta_t)
                    h_adv = torch.stack(h_adv)
                    h_adv = h_adv.permute(1, 0, 2, 3)
                    h_adv = h_adv.detach().cpu().numpy()
                    h_list.append(h_adv)

                x_all = np.concatenate(x_list)
                h_all = np.concatenate(h_list)
                path = './cluster/' + task + '/'
                if not os.path.exists(path):
                    os.makedirs(path)
                path_full = path + subtask + '_length_' + str(self.args.predict_length) + '_subject_' + str(self.args.test_num) + '.mat'
                savemat(path_full,
                        {'org_data': x_all, 'enc_data': h_all})

    def theta(self):
        model_path = './model/example.pth'
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        task = self.args.task_dict
        surrogate_dict = ['no', 'phase', 'time']
        for surrogate in surrogate_dict:
            test_loader_struct, edge, value, _ = load_fmri_test(self.args.start_time,
                                                                self.args.predict_length,
                                                                self.args.delay, self.args.batch_size_test,
                                                                task, self.args.test_num, self.args.thred, surrogate=surrogate)
            t_all_evo = []
            for batch_idx, struct in enumerate(test_loader_struct):
                data_struct, edge, value = struct.to(self.device), edge.to(self.device), value.to(self.device)
                t_evo = self.model.theta(data_struct, edge, value, self.args.predict_length, self.args.hidden_dim)
                t_all_evo.append(t_evo.detach().cpu().numpy())
            t_all_evo = np.concatenate(t_all_evo, axis=1)

            path = './theta/'
            os.makedirs(path, exist_ok=True)
            if surrogate == 'phase':
                path_full = path + task + '_phase.mat'
                theta_name = 'the_phase'
                savemat(path_full, {theta_name: t_all_evo})
            elif surrogate == 'time':
                path_full = path + task + '_time.mat'
                theta_name = 'the_time'
                savemat(path_full, {theta_name: t_all_evo})
            else:
                path_full = path + task + '_subject_' + str(self.args.test_num) + '.mat'
                theta_name = 'the'
                savemat(path_full, {theta_name: t_all_evo})

    def rest(self):
        model_path = './model/example.pth'
        print(model_path)
        self.model.load_state_dict(torch.load(model_path))
        task = self.args.task_dict
        test_loader_struct, edge, value, _ = load_fmri_test(self.args.start_time,
                                                            self.args.predict_length,
                                                            self.args.delay, self.args.batch_size_test,
                                                            task, self.args.test_num, self.args.thred)

        grad_all_evo = []
        for batch_idx, struct in enumerate(test_loader_struct):
            data_struct, edge, value = struct.to(self.device), edge.to(self.device), value.to(self.device)
            grad_evo = self.model.rest(data_struct, edge, value, self.args.predict_length, self.args.hidden_dim)
            grad_all_evo.append(grad_evo.detach().cpu().numpy())
        grad_all_evo = np.concatenate(grad_all_evo, axis=1)

        path = './rest/'
        os.makedirs(path, exist_ok=True)
        path_full = path + 'length_' + str(self.args.predict_length) + '_subject_' + str(self.args.test_num) + '.mat'
        grad_name = 'grad'
        import hdf5storage
        hdf5storage.savemat(path_full,
                            {grad_name: grad_all_evo},
                            format='7.3',
                            compress=True,
                            truncate_existing=True)

    def task(self):
        torch.cuda.empty_cache()
        for timerow_idx in range(len(self.args.timerow)):
            timerow = self.args.timerow[timerow_idx]
            model_path = './model/example.pth'
            print(model_path)
            self.model.load_state_dict(torch.load(model_path))
            for subtask_idx in range(len(self.args.subtask_dict)):
                task = self.args.task_dict
                subtask = self.args.subtask_dict[subtask_idx]

                need_compare = False
                if (subtask == "present" or subtask == "t") and timerow != "time1":
                    need_compare = True
                print(timerow, subtask, need_compare)
                test_loader_struct, edge, value, adj, ids = load_fmri_time(self.args.predict_length,
                                                                      self.args.delay, self.args.awake_delay,
                                                                      self.args.batch_size_test, task, subtask,
                                                                      self.args.test_num, timerow, need_compare, self.args.thred)
                grad_all_in = []
                grad_all_out = []
                x_all = []
                for data, sub in test_loader_struct:
                    x_all.append(data)
                    data_struct, edge, value = data.to(self.device), edge.to(self.device), value.to(self.device)
                    grad_out, grad_in = self.model.task(data_struct, edge, value, self.args.predict_length,
                                                             self.args.delay, self.args.awake_delay, need_compare,
                                                             self.args.hidden_dim)
                    grad_all_in.append(grad_in)
                    grad_all_out.append(grad_out)
                grad_all_in = np.concatenate(grad_all_in, 1)
                grad_all_out = np.concatenate(grad_all_out, 1)
                if need_compare:
                    path = './task/' + task + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path_full = path + subtask + '_awake_' + str(self.args.awake_delay) + '_length_' + str(
                        self.args.predict_length) + '_' + timerow + '_subject_' + str(self.args.test_num) + '.mat'
                    out_name = 'grad_out'
                    in_name = 'grad_in_' + subtask + '_' + timerow
                    import hdf5storage
                    hdf5storage.savemat(path_full,
                                        {out_name: grad_all_out, in_name: grad_all_in, 'id': ids},
                                        format='7.3',
                                        compress=True,
                                        truncate_existing=True)
                else:
                    path = './task/' + task + '/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    path_full = path + subtask + '_awake_' + str(self.args.awake_delay) + '_length_' + str(
                        self.args.predict_length) + '_' + timerow + '_subject_' + str(self.args.test_num) + '.mat'
                    in_name = 'grad_in_' + subtask + '_' + timerow
                    import hdf5storage
                    hdf5storage.savemat(path_full,
                                        {in_name: grad_all_in, 'id': ids},
                                        format='7.3',
                                        compress=True,
                                        truncate_existing=True)