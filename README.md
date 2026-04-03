# Universal-Brain-Dynamics
For "A Universal Space of Brain Dynamics for Unveiling Cognitive Transitions and Individual Differences"

## Data
The repository does not distribute any raw neuroimaging or third‑party datasets. To reproduce the results, you will need to obtain the data from the original sources and respect their usage policies.

Human Connectome Project (HCP)

UK Biobank imaging data

Neuroimaging data from UK Biobank are available by application via the UK Biobank access management system (project application ID as stated in the manuscript).

# UBD Experiment Guide
Quick start, data should be in the folder 'dataset':

For train model: python run.py --net 0

For test model: python run.py --net 1

This project switches between different experiment modes through the `net` parameter in `run.py`. The current implementation contains only one model class, `Exp_gcn`, and different `net` values correspond to different experiment workflows.

Matlab code should be executed after obtaining data from run.py or downloading from Zenodo https://zenodo.org/records/19394402.


## 1. Entry Point

- Main entry: `python run.py`
- Mode switch: directly modify `net` in `run.py`
- Current default value: `net = 1`

The mapping between `net` and experiment workflows is as follows:

| net | Method | Meaning |
| --- | --- | --- |
| 0 | `exp.train()` | Train the model |
| 1 | `exp.predict()` | Prediction experiment |
| 2 | `exp.classfy()` | Export sequential/shuffled classification features |
| 3 | `exp.cluster()` | Export latent representations for clustering |
| 4 | `exp.theta()` | Generate theta evolution results |
| 5 | `exp.rest()` | Resting-state brain dynamics analysis |
| 6 | `exp.task()` | Task-evoked brain dynamics analysis |

## 2. Default Global Parameters

The following parameters are defined at the top of `run.py` and are used as the default values for argparse.

| Parameter | Default | Description |
| --- | --- | --- |
| `net` | `1` | Experiment mode |
| `delay` | `18` | Time embedding length |
| `size` | `426` | Number of brain regions |
| `time_shift` | `10` | Prediction horizon / temporal unfolding length |
| `hidden_dim` | `18` | Latent dimension |
| `train` | `'res'` | Source task used for training data |
| `task_dict` | `['res']` | Current task list to run |
| `num_dict` | `[35]` | List of subject counts for training |
| `need_compare` | `True` | Whether to compare two conditions |
| `test_num` | `40` | Number of test subjects |
| `predict_length` | `5` | Prediction length for testing |
| `awake_delay` | `0` | Post-event temporal offset |

## 3. Tasks and Subtasks

`run.py` defines mappings between tasks and subtasks.

### 3.1 Task-to-subtask mapping

| task | subtask |
| --- | --- |
| `res` | `['no']` |
| `wm` | `['body0', 'face0', 'tool0', 'place0', 'body2', 'face2', 'tool2', 'place2']` |
| `mot` | `['lf', 'rf', 'lh', 'rh', 't']` |
| `lan` | `['question', 'present', 'response']` |

### 3.2 Task-to-`timerow` mapping

| task | timerow |
| --- | --- |
| `res` | `['no']` |
| `wm` | `['time1']` |
| `mot` | `['time2', 'time1']` |
| `lan` | `['time2', 'time1', 'time3']` |

## 4. argparse Parameter Reference

### 4.1 Model structure parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--is_training` | `net` | Current mode id |
| `--epochs` | `3001` | Number of training epochs |
| `--batch_size` | `256` | Training batch size |
| `--batch_size_test` | `32` | Test batch size |
| `--enc_width` | `[18, 32, 128, 32, 18]` | Encoder layer widths |
| `--dec_width` | `[18, 32, 128, 32, 18]` | Decoder layer widths |
| `--aux_width` | `[7668, 852, 3834]` | Layer widths for theta / DKO branch |
| `--learning_rate` | `1e-4` | Learning rate |
| `--weight_decay` | `5e-4` | Weight decay |

Where:

- `enc_width = [delay, 32, 128, 32, hidden_dim]`
- `dec_width = [hidden_dim, 32, 128, 32, delay]`
- `aux_width = [size * hidden_dim, size * 2, int(size * hidden_dim / 2)]`

### 4.2 Data and experiment parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--train_num_dict` | `[35]` | List of subject counts for training |
| `--thred` | `1` | DTI threshold |
| `--train` | `'res'` | Training task |
| `--time_shifts` | `10` | Number of time steps |
| `--delay` | `18` | Time embedding length |
| `--hidden_dim` | `18` | Latent dimension |
| `--dims` | `426` | Number of brain regions |
| `--delta_t` | `0.72` | Sampling interval |
| `--test_num` | `40` | Number of test subjects |
| `--predict_length` | `5` | Prediction length |
| `--need_compare` | `True` | Whether to compare conditions |
| `--start_time` | `0` | Test start time index |
| `--task_dict` | current `task` | Current task |
| `--subtask_dict` | subtasks of current task | Current subtask list |
| `--awake_delay` | `0` | Post-event delay |
| `--timerow` | timerows of current task | Current timerow list |

### 4.3 Device parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--use_gpu` | `True` | Whether to use GPU |
| `--gpu` | `0` | GPU id |
| `--use_multi_gpu` | `False` | Whether to use multiple GPUs |
| `--devices` | `'0,1,2,3'` | Multi-GPU device list |

## 5. Experiment Modes by `net`

### net = 0 Train the model

- Method called: `Exp_gcn.train()`
- Main inputs:
  - `./dataset/{train}.npz`
  - `./dataset/dti.npz`
- Data range:
  - Uses `struct_data`
  - During training, slices `x_struct[200:1000, :train_num, :]`
- Key parameters:
  - `epochs`
  - `batch_size`
  - `train_num_dict`
  - `train`
  - `thred`
  - `time_shifts`
  - `delay`
  - `hidden_dim`
- Loss functions:
  - `loss_pred = MSE(x_pred, dat)`
  - `loss_lin = MSE(y_list, p_list)`
  - Total loss: `loss = loss_pred + loss_lin`
- Model saving:
  - Saves a checkpoint every 100 epochs
  - Output directory: `./model/train_on_{train}_hidden_{hidden_dim}/num_{train_num}_thred_{thred}/`

### net = 1 Prediction experiment

- Method called: `Exp_gcn.predict()`
- Model loaded from: `./model/example.pth`
- Main inputs:
  - `fmri_data` from `./dataset/{task}.npz`
  - `./dataset/dti.npz`
- Key parameters:
  - `start_time`
  - `predict_length`
  - `delay`
  - `test_num`
  - `thred`
  - `hidden_dim`
- Output directory: `./predict/`
- Output file: `{task}_subject_{test_num}.mat`
- Output content:
  - `t`: ground truth
  - `x_adv`: evolved prediction results
  - `x_enc`: encoder-path results

### net = 2 Classification feature export

- Method called: `Exp_gcn.classfy()`
- Model loaded from: `./model/example.pth`
- Main purpose:
  - Extracts random time slices and continuous time slices
  - Exports raw signals and encoded signals for downstream classification or representation analysis
- Fixed parameters in code:
  - `random_time = 100`
  - `st = 20`
- Key parameters:
  - `predict_length`
  - `delay`
  - `test_num`
  - `thred`
- Output directory: `./classfy/subject_{test_num}_length_{random_time}/`
- Output file: `{task}_subject_{test_num}.mat`
- Output content:
  - `x_shu_{task}`
  - `x_seq_{task}`
  - `e_shu_{task}`
  - `e_seq_{task}`

### net = 3 Clustering representation export

- Method called: `Exp_gcn.cluster()`
- Model loaded from: `./model/example.pth`
- Main purpose:
  - Exports raw temporal windows and latent representations for each `timerow` and `subtask`
- Key parameters:
  - `predict_length`
  - `delay`
  - `awake_delay`
  - `test_num`
  - `timerow`
  - `subtask_dict`
- Output directory: `./cluster/{task}/`
- Output file: `{subtask}_length_{predict_length}_subject_{test_num}.mat`
- Output content:
  - `org_data`
  - `enc_data`

### net = 4 Generate theta

- Method called: `Exp_gcn.theta()`
- Model loaded from: `./model/example.pth`
- Main purpose:
  - Generates theta evolution results for original data, phase-randomized surrogate data, and time-randomized surrogate data
- Surrogate types:
  - `no`
  - `phase`
  - `time`
- Key parameters:
  - `predict_length`
  - `delay`
  - `test_num`
  - `thred`
  - `hidden_dim`
- Output directory: `./theta/`
- Output files and variable names:
  - Original data: `{task}_subject_{test_num}.mat`, variable name `the`
  - Phase surrogate: `{task}_phase.mat`, variable name `the_phase`
  - Time surrogate: `{task}_time.mat`, variable name `the_time`

### net = 5 Resting-state brain dynamics

- Method called: `Exp_gcn.rest()`
- Model loaded from: `./model/example.pth`
- Main purpose:
  - Computes gradient propagation results on resting-state sequences
- Key parameters:
  - `predict_length`
  - `delay`
  - `test_num`
  - `thred`
- Output directory: `./rest/`
- Output file: `length_{predict_length}_subject_{test_num}.mat`
- Output variable:
  - `grad`
- Storage format:
  - Uses `hdf5storage.savemat(..., format='7.3')`

### net = 6 Task-evoked brain dynamics

- Method called: `Exp_gcn.task()`
- Model loaded from: `./model/example.pth`
- Main purpose:
  - Computes task-evoked gradient propagation results for different task time blocks and subtasks
- `need_compare` logic:
  - Automatically set to `True` when `subtask == 'present'` or `subtask == 't'`, and `timerow != 'time1'`
  - Otherwise set to `False`
- Key parameters:
  - `predict_length`
  - `delay`
  - `awake_delay`
  - `test_num`
  - `timerow`
  - `subtask_dict`
  - `hidden_dim`
  - `thred`
- Output directory: `./task/{task}/`
- Output file:
  - `{subtask}_awake_{awake_delay}_length_{predict_length}_{timerow}_subject_{test_num}.mat`
- Output variables:
  - When `need_compare=True`:
    - `grad_out`
    - `grad_in_{subtask}_{timerow}`
    - `id`
  - When `need_compare=False`:
    - `grad_in_{subtask}_{timerow}`
    - `id`

## 6. Common Usage

This project is mainly executed by modifying the variables at the top of `run.py`, rather than passing all parameters externally through the command line.

### 6.1 Train the model

1. Set `net = 0` in `run.py`
2. For example, set:
   - `train = 'res'`
   - `task_dict = ['res']`
   - `num_dict = [35]`
3. Run:

```bash
python run.py
```

### 6.2 Run the prediction experiment

1. Set `net = 1`
2. Make sure `./model/example.pth` exists
3. Set:
   - `task_dict = ['res']`
   - `test_num = 40`
   - `predict_length = 5`
4. Run:

```bash
python run.py
```

### 6.3 Run task-state analysis

1. Set `net = 6`
2. For example, set:
   - `task_dict = ['lan']` or `['mot']`
   - `predict_length = 5`
   - `awake_delay = 0`
3. Run:

```bash
python run.py
```

## 7. Output Directory Summary

| net | Output directory |
| --- | --- |
| 0 | `./model/` |
| 1 | `./predict/` |
| 2 | `./classfy/` |
| 3 | `./cluster/` |
| 4 | `./theta/` |
| 5 | `./rest/` |
| 6 | `./task/` |

## 8. Notes

- For `net = 1 ~ 6`, the code loads the model from `./model/example.pth` by default. After training, you usually need to manually copy or rename a checkpoint to `example.pth`, or directly modify the model path in code.
- In `load_fmri_time()`, the data path is hard-coded as `D:/github/gcn_np/dataset/{task}.npz`. If your current environment does not use this path, you need to modify the code first.
- `load_fmri_train()` reads `struct_data`, while `load_fmri_test()` reads `fmri_data`. Before running experiments, make sure the corresponding fields exist in the `.npz` files.
- `classfy()`, `cluster()`, `theta()`, `rest()`, and `task()` are essentially result-export experiments and do not retrain the model.
