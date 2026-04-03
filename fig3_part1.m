%% classify to eight states in Fig.3a
clearvars; clc
addpath("matlab_code")
data_types = {
    {'e_seq_emo', 'e_seq_gam', 'e_seq_lan', 'e_seq_mot', 'e_seq_rel', 'e_seq_res', 'e_seq_soc', 'e_seq_wm'}, 'e_seq';
    {'e_shu_emo', 'e_shu_gam', 'e_shu_lan', 'e_shu_mot', 'e_shu_rel', 'e_shu_res', 'e_shu_soc', 'e_shu_wm'}, 'e_shu';
    {'x_seq_emo', 'x_seq_gam', 'x_seq_lan', 'x_seq_mot', 'x_seq_rel', 'x_seq_res', 'x_seq_soc', 'x_seq_wm'}, 'x_seq';
    {'x_shu_emo', 'x_shu_gam', 'x_shu_lan', 'x_shu_mot', 'x_shu_rel', 'x_shu_res', 'x_shu_soc', 'x_shu_wm'}, 'x_shu'
};
class_names = {'emo', 'gam', 'lan', 'mot', 'rel', 'res', 'soc', 'wm'};
folder_path = 'classify';

fprintf('load task data...\n');
for i = 1:length(class_names)
    file_path = fullfile(folder_path, class_names{i});
    fprintf('  loaded: %s\n', class_names{i});
    load(file_path);
end

sub_values = 10:10:50;  % set subject number
t_values = 10:10:100;   % set temporal sample

n_subs = length(sub_values);
n_times = length(t_values);
n_data_types = size(data_types, 1);

accuracy_results = zeros(n_subs, n_times, n_data_types);

for sub_idx = 1:n_subs
    sub = sub_values(sub_idx);
    
    for t_idx = 1:n_times
        t = t_values(t_idx);
        for data_type_idx = 1:size(data_types, 1)
            var_names = data_types{data_type_idx, 1};
            model_name = data_types{data_type_idx, 2};    
            fprintf('\n--- train %s model ---\n', model_name);
            
            data_cell = cell(1, 8);
            data_exists = true;
            
            for i = 1:8
                if exist(var_names{i}, 'var')
                    data_cell{i} = eval(var_names{i});
                else
                    fprintf('variable %s does not exist\n', var_names{i});
                    data_exists = false;
                    break;
                end
            end
            
            if ~data_exists
                fprintf('skip %s model training\n', model_name);
                continue;
            end
            
            for i = 1:length(data_cell)
                data_cell{i} = data_cell{i}(sub-9:sub, 1:t, :, :);
                sz = size(data_cell{i});
                data_cell{i} = reshape(data_cell{i}, [sz(1)*sz(2),sz(3)*sz(4)]);
            end
            
            X = vertcat(data_cell{:});
            Y = cell2mat(arrayfun(@(i) i*ones(size(data_cell{i},1), 1), ...
                                   1:length(data_cell), 'UniformOutput', false)');
            
            fprintf('\n combining to: X=%s, Y=%s\n', mat2str(size(X)), mat2str(size(Y)));
            
            rng(1); % set random seed
            cv = cvpartition(Y, 'HoldOut', 0.3); % train-test partition
            
            X_train = X(training(cv), :);
            Y_train = Y(training(cv));
            X_test = X(test(cv), :);
            Y_test = Y(test(cv));
            fprintf('train-test partition:\n');
            fprintf('  training set: %d samples\n', size(X_train, 1));
            
            fprintf('start train SVM model...\n');
            svm_model = fitcecoc(X_train, Y_train, ...
                'Learners', templateSVM('KernelFunction', 'gaussian'), 'Coding', 'onevsall');
        
            Y_test_pred = predict(svm_model, X_test);
            test_accuracy = sum(Y_test_pred == Y_test) / length(Y_test) * 100;
            fprintf('accuaracy in test set: %.2f%%\n', test_accuracy);
            accuracy_results(sub_idx, t_idx, data_type_idx) = test_accuracy;
        end
    end
end

accuracy_mean = squeeze(mean(accuracy_results, 1));  
accuracy_std = squeeze(std(accuracy_results, 0, 1)); 
hold on;
markers = {'o', 's', 'd', '>'};

color1 = [0.017, 0.443, 0.690];       
color2 = [0.843, 0.098, 0.110];     
color3 = [0.106, 0.471, 0.216];   
color4 = [0.318, 0.118, 0.510];   

colors = [color1; color2; color3; color4];

for data_type_idx = 1:n_data_types
    y_mean = accuracy_mean(:, data_type_idx);
    y_std = accuracy_std(:, data_type_idx);
    
    fill([t_values, fliplr(t_values)], ...
        [(y_mean + y_std)', fliplr((y_mean - y_std)')], ...
        colors(data_type_idx, :), 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
end

plot_handles = zeros(n_data_types, 1);
for data_type_idx = 1:n_data_types
    y_mean = accuracy_mean(:, data_type_idx);

    c = colors(data_type_idx, :);
    if data_type_idx >= 3   
        c = [c 0.3];        
    end

    plot_handles(data_type_idx) = plot(t_values, y_mean, ...
        ['-' markers{mod(data_type_idx-1, length(markers))+1}], ...
        'LineWidth', 2, 'MarkerSize', 6, ...
        'MarkerFaceColor', colors(data_type_idx, :), ...
        'Color', c);
end

hold off;

legend(["Sequential latent" "Shuffled latent" "Sequential fMRI" "Shuffled fMRI"], ...
    'Location', 'best', 'FontSize', 9.4);
grid on;
xlim([min(t_values)-1, max(t_values)+1]);
ylim([9, 100]);
ax = gca;
ax.FontSize = 11;
set(gcf, 'Units', 'inches', 'Position', [0, 0, 3.9, 3]);
xlabel('Samples', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontSize', 13);