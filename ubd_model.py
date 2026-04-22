from utils import *

class GraphConv(torch.nn.Module):
    def __init__(self, args, enc_width, dec_width, aux_width):
        print('Now use GraphConv')
        super().__init__()
        self.size = args.dims
        self.delay = args.delay
        self.encoder = MLN_GraphConv(enc_width)
        self.decoder = MLN_GraphConv(dec_width)
        self.the = MLN_MLP(aux_width)

    def forward(self, data, edge, value, time_shifts, hidden_dim):
        y_list = []
        x_list = []
        p_list = []

        self.data = torch.transpose(data, 0, 1)
        y_advanced = self.encoder(self.data[0], edge, value)
        y_advanced = y_advanced.view(y_advanced.shape[0], -1)
        for j in range(time_shifts):
            if j > 0 and j <= time_shifts-1:
                x = self.data[j]
                y_temp = self.encoder(x, edge, value)
                y_list.append(y_temp)

            the = y_advanced
            the = self.the(the)
            y_advanced = varying_multiply1(y_advanced, the, 0.72)
            temp = y_advanced.view(y_advanced.shape[0], self.size, hidden_dim)
            x_pred = self.decoder(temp, edge, value)
            x_list.append(x_pred)
            p_list.append(temp)
        p_list = p_list[:-1]
        x_list = torch.stack(x_list)
        return self.data[1:], x_list, p_list, y_list

    def predict(self, data, edge, value, time_shifts, hidden_dim):
        x1_list = []
        x2_list = []
        # data = data[:, :, torch.randperm(data.shape[2])]
        y_adv = self.encoder(data[:, 0], edge, value)
        y_adv = y_adv.view(y_adv.shape[0], -1)

        for j in range(1, time_shifts+1):
            y_enc = self.encoder(data[:, j], edge, value)
            y_enc = y_enc.view(y_enc.shape[0], -1)
            the_enc = self.the(y_enc)
            the = self.the(y_adv)
            y_adv = varying_multiply1(y_adv, the, 0.72)
            y_enc = varying_multiply1(y_enc, the_enc, 0.72)
            temp1 = y_adv.view(y_adv.shape[0], self.size, hidden_dim)
            temp2 = y_enc.view(y_enc.shape[0], self.size, hidden_dim)
            x_pred1 = self.decoder(temp1, edge, value)
            x_pred2 = self.decoder(temp2, edge, value)
            x1_list.append(x_pred1[:, :, 0])
            x2_list.append(x_pred2[:, :, 0])
        x1_list = torch.stack(x1_list).permute(1, 0, 2)
        x2_list = torch.stack(x2_list).permute(1, 0, 2)
        return x1_list, x2_list

    def classfy(self, data_shu, data_seq, edge, value):
        e_shu = self.encoder(data_shu, edge, value)
        e_seq = self.encoder(data_seq, edge, value)
        # return data_shu[:, :, :, 0], data_seq[:, :, :, 0], e_shu[:, :, :, 0], e_seq[:, :, :, 0]
        return data_shu, data_seq, e_shu, e_seq

    def theta(self, data, edge, value, time_shifts, hidden_dim):
        the_list_evo = []
        head = data[:, 0]
        y_advanced = self.encoder(head, edge, value)
        for j in range(time_shifts):
            # y_advanced = self.encoder(data[:, j], edge, value)
            y_advanced = y_advanced.view(y_advanced.shape[0], -1)
            the_evo = self.the(y_advanced)
            y_advanced = varying_multiply1(y_advanced, the_evo, 0.72)
            the_evo = the_evo.view(y_advanced.shape[0], self.size, int(hidden_dim / 2))
            y_advanced = y_advanced.view(y_advanced.shape[0], self.size, hidden_dim)
            the_list_evo.append(the_evo)

        the_list_evo = torch.stack(the_list_evo)
        return the_list_evo

    def rest(self, data, edge, value, time_shifts, delta_t):
        data.requires_grad_(True)
        head = data[:, 0]
        y = self.encoder(head, edge, value)
        y_list = []
        grad_time = []
        for j in range(time_shifts):
            grad = torch.zeros((data.shape[0], data.shape[2], data.shape[2], data.shape[3]),
                               device=data.device)
            y_list.append(y)
            y = y.view(y.shape[0], -1)
            the = y
            the = self.the(the)
            y = varying_multiply1(y, the, delta_t)
            y = y.view(y.shape[0], self.size, self.delay)
            x_pred = self.decoder(y, edge, value)
            for i in range(data.shape[2]):
                grad_temp = torch.autograd.grad(outputs=x_pred[:, i],
                                                inputs=head,
                                                grad_outputs=torch.ones_like(x_pred[:, i]),
                                                retain_graph=True)[0]
                grad[:, i, :, :] = grad_temp
            grad_time.append(grad)
        grad_all = torch.stack(grad_time).mean(dim=-1)
        return grad_all

    def task(self, data, edge, value, time_shifts, delay, awake_delay, need_compare, hidden_dim):
        if need_compare:
            head_out, head_in = data[:, 0], data[:, (delay+4+awake_delay)]
            head_out.requires_grad_(True)
            head_in.requires_grad_(True)
            y_out = self.encoder(head_out, edge, value)
            y_in = self.encoder(head_in, edge, value)
            y_out = y_out.view(y_out.shape[0], -1)
            y_in = y_in.view(y_in.shape[0], -1)

            grad_in_list = []
            grad_out_list = []

            for j in range(1):
                the_out = self.the(y_out)
                y_out = varying_multiply1(y_out, the_out, 0.72)
                grad_out = torch.zeros((data.shape[0], data.shape[2], data.shape[2], data.shape[3]),
                                       device=data.device, dtype=torch.float32)
                y_out_temp = y_out.view(y_out.shape[0], self.size, hidden_dim)
                for i in range(data.shape[2]):
                    grad_temp = torch.autograd.grad(outputs=y_out_temp[:, i],
                                                    inputs=head_out,
                                                    grad_outputs=torch.ones_like(y_out_temp[:, i]),
                                                    retain_graph=True)[0]
                    grad_out[:, i, :, :] = grad_temp
                grad_out_list.append(grad_out.cpu())

            for j in range(time_shifts):

                the_in = self.the(y_in)
                y_in = varying_multiply1(y_in, the_in, 0.72)
                grad_in = torch.zeros((data.shape[0], data.shape[2], data.shape[2], data.shape[3]),
                                      device=data.device, dtype=torch.float32)
                y_in_temp = y_in.view(y_in.shape[0], self.size, hidden_dim)
                for i in range(data.shape[2]):
                    grad_temp = torch.autograd.grad(outputs=y_in_temp[:, i],
                                                    inputs=head_in,
                                                    grad_outputs=torch.ones_like(y_in_temp[:, i]),
                                                    retain_graph=True)[0]
                    grad_in[:, i, :, :] = grad_temp
                grad_in_list.append(grad_in.cpu())

            grad_in = torch.stack(grad_in_list, dim=0).mean(dim=-1)
            grad_out = torch.stack(grad_out_list, dim=0).mean(dim=-1)
            return grad_out, grad_in
        else:
            grad_time = []
            head = data[:, awake_delay]
            head.requires_grad_(True)
            y = self.encoder(head, edge, value)
            for j in range(time_shifts):
                y = y.view(y.shape[0], -1)
                the = y
                the = self.the(the)
                y = varying_multiply1(y, the, 0.72)
                y = y.view(y.shape[0], self.size, hidden_dim)
                grad = torch.zeros((data.shape[0], data.shape[2], data.shape[2], data.shape[3]),
                                   device=data.device)
                for i in range(data.shape[2]):
                    grad_temp = torch.autograd.grad(outputs=y[:, i],
                                                    inputs=head,
                                                    grad_outputs=torch.ones_like(y[:, i]),
                                                    retain_graph=True)[0]
                    grad[:, i, :, :] = grad_temp
                grad_time.append(grad.cpu().detach())
            grad_all = torch.stack(grad_time, dim=0).mean(dim=-1)
            return grad_all, grad_all

    def cluster(self, data, edge, value, time_shifts, delta_t):
        x_list = []
        y_adv = self.encoder(data[:, 0], edge, value)
        y_adv = y_adv.view(y_adv.shape[0], -1)

        for j in range(0, time_shifts):
            temp = y_adv.view(y_adv.shape[0], self.size, self.delay)
            x_list.append(temp)
            the_adv = self.the(y_adv)
            y_adv = varying_multiply1(y_adv, the_adv, delta_t)

        return x_list
