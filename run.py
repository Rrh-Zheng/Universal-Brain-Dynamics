import argparse
import torch
import utils
from exp.train_gcn import Exp_gcn

task_subtask_dict = {
    'wm': ['body0', 'face0', 'tool0', 'place0', 'body2', 'face2', 'tool2', 'place2'],
    'res': ['no'],
    'mot': ['lf', 'rf', 'lh', 'rh', 't'],
    'lan': ['question', 'present', 'response'],
}

subtask_time_dict = {
    'wm': ['time1'],
    'res': ['no'],
    'mot': ['time2', 'time1'],
    'lan': ['time2', 'time1', 'time3']
}

###
# use 'net' control model
# 0 = train model
# 1 = predict
# 2 = classify
# 3 = clustering
# 4 = generate theta
# 5 = brain dynamics in resting-state
# 6 = brain dynamics in task-evoked state
net = 1
###
delay = 18
size = 426
time_shift = 10
hidden_dim = 18
train = 'res'
# task_dict = ['res', 'wm', 'mot', 'lan', 'emo', 'rel', 'gam', 'soc']
task_dict = ['res']

for i in range(len(task_dict)):
    task = task_dict[i]
    if task in ('mot', 'lan', 'wm'):
        subtask = task_subtask_dict[task]
    else:
        subtask = task_subtask_dict['res']
    num_dict = [35]
    need_compare = True
    test_num = 40
    predict_length = 5
    awake_delay = 0
    timerow = subtask_time_dict[task]

    parser = argparse.ArgumentParser()

    # Setting main model parameters
    parser.add_argument('--is_training', type=int, required=False, default=net, help='status')
    parser.add_argument('--epochs', type=int, default=3001, help='training epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='test batch size')
    parser.add_argument('--enc_width', type=list, default=[delay, 32, 128, 32, hidden_dim], help='encoder size')
    parser.add_argument('--dec_width', type=list, default=[hidden_dim, 32, 128, 32, delay], help='decoder size')
    parser.add_argument('--aux_width', type=list, default=[size*hidden_dim, size*2, int(size*hidden_dim/2)], help='DKO size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')

    # Setting train detail
    parser.add_argument('--train_num_dict', type=int, nargs='+', default=num_dict, help='subject number used for training')
    parser.add_argument('--thred', type=int, default=1, help='thredshold of DTI')
    parser.add_argument('--train', type=str, default=train, help='state of train data sample from (default as rest)')
    parser.add_argument('--time_shifts', type=int, default=time_shift, help='length of prediction loss')
    parser.add_argument('--delay', type=int, default=delay, help='time embedding delay')
    parser.add_argument('--hidden_dim', type=int, default=hidden_dim, help='size of latent representation (default as time delay embeding)')
    parser.add_argument('--dims', type=int, default=size, help='number of brain areas')
    parser.add_argument('--delta_t', type=int, default=0.72, help='delta t (default as 0.72)')

    # Setting test detail
    parser.add_argument('--test_num', type=int, default=test_num, help='subject number used for test')
    parser.add_argument('--predict_length', type=int, default=predict_length, help='test prediction length')
    parser.add_argument('--need_compare', type=bool, default=need_compare, help='need compare two conditions')
    parser.add_argument('--start_time', type=int, default=0, help='choose start time point')
    parser.add_argument('--task_dict', type=str, nargs='+', default=task, help='assign task')
    parser.add_argument('--subtask_dict', type=str, nargs='+', default=subtask, help='assign subtask')
    parser.add_argument('--awake_delay', type=int, default=awake_delay, help='assign time point after recorded time')
    parser.add_argument('--timerow', type=str, nargs='+', default=timerow, help='assign time block')

    # Setting equipment detail
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    utils.setup_seed(7)


    exp = Exp_gcn(args)

    if args.is_training == 0:
        exp.train()
    elif args.is_training == 1:
        exp.predict()
    elif args.is_training == 2:
        exp.classfy()
    elif args.is_training == 3:
        exp.cluster()
    elif args.is_training == 4:
        exp.theta()
    elif args.is_training == 5:
        exp.rest()
    elif args.is_training == 6:
        exp.task()