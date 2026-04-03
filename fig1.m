%% draw correlation along with time of all subject in one task
addpath("matlab_code")
predict_length = 100;
start = 1;
time_roll = 18;
figure; hold on;
[cor_subjects1, cor_mean] = plot_correlation(x_adv, t, predict_length, start);
for sub = 1:size(t,1)
    plot(cor_subjects1(sub,:), 'Color', [0.5, 0.5, 0.5, 0.3]);
end

plot(cor_mean, 'k', 'LineWidth', 2); 
cor_mean_full = mean(cor_mean,"all", 'omitmissing');
xlim([5 predict_length])
xline(time_roll - start, 'k', 'LineWidth', 1);
xlabel('Predict time')
ylabel('Correlation') 
title(['Mean of correlation: ', num2str(cor_mean_full), ' Number of subjects:', num2str(sub)], 'FontSize', 14)
[order_map, class_label, area_name, specify] = template_info();
%% draw all tasks in Fig.1d
% make sure data in folder 'predict' exsit (created by run.py)
folder_path = 'predict';
mat_files = dir(fullfile(folder_path, '*.mat'));

data_struct = struct();

for i = 1:length(mat_files)
    filename = mat_files(i).name;
    [~, name, ~] = fileparts(filename);  
    
    file_path = fullfile(folder_path, filename);
    loaded_data = load(file_path);
    
    field_names = fieldnames(loaded_data);
    for j = 1:length(field_names)
        original_var = field_names{j};
        new_var_name = sprintf('%s_%s', original_var, name);
        
        assignin('base', new_var_name, loaded_data.(original_var));
        
        data_struct.(new_var_name) = loaded_data.(original_var);
    end
    
    fprintf('loaded: %s\n', filename);
end

fprintf('data loaded\n');

dataset_names = {'emo', 'gam', 'lan', 'mot', 'rel', 'res', 'soc', 'wm'};
num_datasets = length(dataset_names);

all_cor_adv = cell(num_datasets, 1);
all_cor_enc = cell(num_datasets, 1);

for dataset_idx = 1:num_datasets
    dataset_name = dataset_names{dataset_idx};
    
    t_var_name = sprintf('t_%s', dataset_name);
    x_adv_var_name = sprintf('x_adv_%s', dataset_name);
    x_enc_var_name = sprintf('x_enc_%s', dataset_name);
    
    if ~exist(t_var_name, 'var') || ~exist(x_adv_var_name, 'var') || ~exist(x_enc_var_name, 'var')
        fprintf('task %s does not exist，skip\n', dataset_name);
        continue;
    end
    
    t = eval(t_var_name);
    x_adv = eval(x_adv_var_name);
    x_enc = eval(x_enc_var_name);
    
    predict_length = size(t, 2);
    num_subjects = size(t, 1);
    idx = 1;
    start = 25;
    row_num = predict_length / start;
    
    cor_all_adv = zeros(row_num, num_subjects, 1);
    cor_all_enc = zeros(row_num, num_subjects, 1);
    
    for time = start:start:predict_length
        cor_sub_enc = zeros(num_subjects, 1);
        cor_sub_adv = zeros(num_subjects, 1);
        
        for sub = 1:num_subjects
            T_adv = squeeze(t(sub, 1:time, :));
            adv = squeeze(x_adv(sub, 1:time, :));
            T_enc = squeeze(t(sub, time-start+1:time, :));
            enc = squeeze(x_enc(sub, time-start+1:time, :));
            
            cor_adv = diag(corr(T_adv, adv));
            cor_enc = diag(corr(T_enc, enc));
            
            cor_sub_adv(sub, :) = mean(cor_adv, "all");
            cor_sub_enc(sub, :) = mean(cor_enc, "all");
        end
        
        cor_all_adv(idx, :, :) = cor_sub_adv;
        cor_all_enc(idx, :, :) = cor_sub_enc;
        idx = idx + 1;
    end
    
    all_cor_adv{dataset_idx} = cor_all_adv;
    all_cor_enc{dataset_idx} = cor_all_enc;
    
    assignin('base', sprintf('cor_all_adv_%s', dataset_name), cor_all_adv);
    assignin('base', sprintf('cor_all_enc_%s', dataset_name), cor_all_enc);
    
    fprintf('calculated: %s\n', dataset_name);
end

fprintf('correlation calculated\n');
group_names = {'rel', 'wm', 'mot', 'lan', 'emo', 'rel', 'gam', 'soc'};
n_data_cols = 4;

task_colors = {
    [0.017, 0.443, 0.690],   
    [0.843, 0.098, 0.110],   
    [0.106, 0.471, 0.216],   
    [0.851, 0.373, 0.008],   
    [0.318, 0.118, 0.510],  
    [0.55 0.34 0.29],   
    [0.89 0.47 0.76],   
    [0.50 0.50 0.50]   
};

figure;
all_data_adv = [];
all_data_enc = [];

for data_col = 1:n_data_cols
    for group_idx = 1:length(group_names)
        group_name = group_names{group_idx};
        
        eval(sprintf('current_data_enc = cor_all_enc_%s;', group_name));
        eval(sprintf('current_data_adv = cor_all_adv_%s;', group_name));
        
        col_data_enc = current_data_enc(data_col, :);
        col_data_adv = current_data_adv(data_col, :);
        
        all_data_adv = [all_data_adv, col_data_adv(:)];
        all_data_enc = [all_data_enc, col_data_enc(:)];
    end
end

hold on;
v_adv = violinplot(all_data_adv);
v_enc = violinplot(all_data_enc);

position = 1;
for data_col = 1:n_data_cols
    for group_idx = 1:length(group_names)
        current_color = task_colors{group_idx};
        
        try
            set(v_adv(position), 'FaceColor', current_color, 'FaceAlpha', 0.3, ...
                 'EdgeColor', current_color, 'LineWidth', 1);
            set(v_enc(position), 'FaceColor', current_color, 'FaceAlpha', 0.9, ...
                'EdgeColor', 'none', 'LineWidth', 0);
        catch
            try
                v_adv(position).ViolinColor = current_color;
                v_adv(position).ViolinAlpha = 0.3;
                v_enc(position).ViolinColor = current_color;
                v_enc(position).ViolinAlpha = 0.9;
            catch
            end
        end
        
        position = position + 1;
    end
end

for data_col = 1:n_data_cols-1
    line_pos = data_col * length(group_names) + 0.5;
    line([line_pos, line_pos], [-0.1, 1], 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 0.5);
end

legend_handles = [];

custom_labels = {'Rest', 'WM', 'Motor', 'Language','Emotion', 'Relational', 'Gambling', 'Social'};

for i = 1:length(group_names)
    color = task_colors{i};
    h = plot(NaN, NaN, 'Color', color, 'LineWidth', 4);
    legend_handles = [legend_handles, h];
end

lg = legend(legend_handles, custom_labels, 'Location', 'southwest', 'FontSize', 20);
lg.Title.String = 'States';

set(gca, 'XTick', []);

time_labels_first = {'1~25', '26~50', '51~75', '76~100'};
x_positions = [0.225, 0.425, 0.625, 0.825]; 

for i = 1:length(time_labels_first)
    annotation('textbox', [x_positions(i)-0.05, 0.05, 0.1, 0.05], ...
        'String', time_labels_first{i}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontSize', 22, ...
        'EdgeColor', 'none');
end

annotation('textbox', [0.35, 0.01, 0.3, 0.04], ...
    'String', 'Time', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'FontSize', 25, ...
    'EdgeColor', 'none');
ylabel('PCC', 'FontSize', 22);

ax = gca;
ax.LineWidth = 1;
ax.Box = 'on';
ax.TickDir = 'out';
ax.FontSize = 22;
ylim([-0.1, 1]);
hold off;

set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1, 1, 10.5, 8]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10, 6]);
set(gcf, 'PaperPosition', [0, 0, 10, 6]);
%% draw whole brain predictive results for specific subject in Fig.1c
subject = 7;
predict_length = 100;

truth = squeeze(t(subject,:,:));
predict = squeeze(x_adv(subject,:,:));
for j = 2:predict_length
    T = truth(1:j, :);
    X = predict(1:j, :);
    cor = diag(corr(T,X));
    cor_all(j,:) = cor;
end
cor_all = abs(cor_all);
figure('Position', [100, 100, 600, 275]);
imagesc(cor_all(1:predict_length,:)');
xline(17, 'k--', 'LineWidth', 1.5);
colorbar
xlabel('Time', 'Fontsize', 16)
ylabel('Area', 'Fontsize', 16)
c = colorbar;
c.Label.FontSize = 15;
n = 256;
red   = [0.843, 0.098, 0.110];  
blue  = [0.017, 0.443, 0.690];   
white = [1,     1,     1    ];

half = n / 2;

CM = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];
colormap(CM)
xlim([10 100])
ax = gca;
ax.FontSize = 16;
%% draw specific area for aboved subject in Fig.1b
area = 168;
figure('Position', [100, 100, 600, 275]);
plot(predict(:,area), 'LineWidth', 2,'Color',[0.843, 0.098, 0.110]);
hold on
plot(truth(:,area), 'LineWidth', 2,'Color',[0.017, 0.443, 0.690]);
xline(17, 'k--', 'LineWidth', 1.5);
legend('Prediction', 'Observation','Fontsize', 15, 'Location', 'best')
xlabel('Time', 'Fontsize', 14)
ylabel('Value', 'Fontsize', 14)
ax = gca;
ax.FontSize = 16;