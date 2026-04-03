%% compare clustring accuracy between latent and fMRI data in Fig.4a
addpath('matlab_code')
base_path = 'cluster\wm\';
csv_file = 'task\wm\wm.xlsx';

sub_inform = readtable(csv_file);
first_col_name = sub_inform.Properties.VariableNames{1}; 
categories = {'body', 'face', 'tool', 'place'};
suffixes = {'0', '2'};
n_files = length(categories) * length(suffixes);

t_values = 25; % time length
s_values = 10:10:100; % subject number

accuracy_enc_all_files = zeros(n_files, length(s_values));
accuracy_org_all_files = zeros(n_files, length(s_values));
file_names_list = cell(n_files, 1);

file_counter = 0; 

for cat_idx = 1:length(categories)
    for suf_idx = 1:length(suffixes)
        
        file_counter = file_counter + 1;
        
        current_cat = categories{cat_idx};
        current_suf = suffixes{suf_idx};
        filename_str = sprintf('%s%s_subject_1000_length_30.mat', current_cat, current_suf);
        file_path = fullfile(base_path, filename_str);
        file_names_list{file_counter} = filename_str;
        
        fprintf('>>> [%d/%d] process: %s\n', file_counter, n_files, filename_str);
        
        data = load(file_path);

        if isfield(data, 'enc_data'), enc_data_raw = data.enc_data; else, error('no enc_data'); end
        if isfield(data, 'org_data'), org_data_raw = data.org_data; else, error('no org_data'); end
        if isfield(data, 'id'), id_raw = data.id; else, error('.mat文件中缺少 id 变量，无法匹配行为分数'); end
        
        [is_match, match_idx] = ismember(id_raw, sub_inform.(first_col_name));
        valid_indices = find(is_match); 
        matched_table_rows = match_idx(valid_indices); 
        
        current_test_values = sub_inform.WM_Task_2bk_Acc(matched_table_rows);

        enc_data_filtered = enc_data_raw(valid_indices, :, :, :);
        org_data_filtered = org_data_raw(valid_indices, :, :, :);
        
        n_matched = length(current_test_values);
        
        [~, sort_idx] = sort(current_test_values);
        
        for idx_s = 1:length(s_values)
            s = s_values(idx_s);
            idx_bottom = sort_idx(1:s);
            idx_top = sort_idx(end-s+1:end);
            t = t_values(1);
            c_top_enc = enc_data_filtered(idx_top, 1:t, :, :);
            c_btm_enc = enc_data_filtered(idx_bottom, 1:t, :, :);
            cluster_enc = cat(1, c_top_enc, c_btm_enc); 
            
            acc_enc = calculate_clustering_accuracy(cluster_enc, size(cluster_enc,1), 1);
            accuracy_enc_all_files(file_counter, idx_s) = acc_enc;
            
            c_top_org = org_data_filtered(idx_top, 1:t, :, :);
            c_btm_org = org_data_filtered(idx_bottom, 1:t, :, :);
            cluster_org = cat(1, c_top_org, c_btm_org);
            
            acc_org = calculate_clustering_accuracy(cluster_org, size(cluster_org,1), 2);
            accuracy_org_all_files(file_counter, idx_s) = acc_org;
            
            fprintf('    s=%d: Enc=%.2f%%, Org=%.2f%%\n', s, acc_enc*100, acc_org*100);
        end
            
        fprintf('\n');
    end
end


function accuracy = calculate_clustering_accuracy(cluster_data, k, method)
    n_samples = size(cluster_data, 1) * size(cluster_data, 2);
    n_features = size(cluster_data, 3) * size(cluster_data, 4);
    
    X = zeros(n_samples, n_features);
    true_labels = zeros(n_samples, 1);
    
    idx = 1;
    for i = 1:size(cluster_data, 1)
        for j = 1:size(cluster_data, 2)
            X(idx, :) = reshape(squeeze(cluster_data(i, j, :, :)), 1, []);
            true_labels(idx) = i;
            idx = idx + 1;
        end
    end
    
    rng(42);
    shuffle_idx = randperm(n_samples);
    X = X(shuffle_idx, :);
    true_labels = true_labels(shuffle_idx);
    
    if method == 1
        predicted_labels = spectralcluster(X, k);
    else
        predicted_labels = kmeans(X, k);
    end
    
    confusion = confusionmat(true_labels, predicted_labels);
    cost_matrix = max(confusion(:)) - confusion; 
    
    [assignment, ~] = matchpairs(cost_matrix, 10000); 
    
    correct = 0;
    for i = 1:size(assignment, 1)
        correct = correct + confusion(assignment(i,1), assignment(i,2));
    end
    accuracy = correct / sum(confusion(:));
end
n_cats = 4;
n_sufs = 2;
n_s_vals = length(s_values); 
data_reshaped_enc = zeros(n_sufs, n_cats, n_s_vals);
data_reshaped_org = zeros(n_sufs, n_cats, n_s_vals);

counter = 0;
for c = 1:n_cats
    for s = 1:n_sufs
        counter = counter + 1;
        % Dim1: Suffix(0/2), Dim2: Category, Dim3: S_values
        data_reshaped_enc(s, c, :) = accuracy_enc_all_files(counter, :);
        data_reshaped_org(s, c, :) = accuracy_org_all_files(counter, :);
    end
end
mu_enc_0 = squeeze(mean(data_reshaped_enc(1, :, :), 2))';
std_enc_0 = squeeze(std(data_reshaped_enc(1, :, :), 0, 2))';
mu_org_0 = squeeze(mean(data_reshaped_org(1, :, :), 2))';
std_org_0 = squeeze(std(data_reshaped_org(1, :, :), 0, 2))';

mu_enc_2 = squeeze(mean(data_reshaped_enc(2, :, :), 2))';
std_enc_2 = squeeze(std(data_reshaped_enc(2, :, :), 0, 2))';
mu_org_2 = squeeze(mean(data_reshaped_org(2, :, :), 2))';
std_org_2 = squeeze(std(data_reshaped_org(2, :, :), 0, 2))';
 
hold on;

color_enc = [0.843, 0.098, 0.110];
color_org = [0.017, 0.443, 0.690];


fill([s_values, fliplr(s_values)], ...
    [(mu_enc_0 + std_enc_0)*100, fliplr((mu_enc_0 - std_enc_0)*100)], ...
    color_enc, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([s_values, fliplr(s_values)], ...
    [(mu_org_0 + std_org_0)*100, fliplr((mu_org_0 - std_org_0)*100)], ...
    color_org, 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([s_values, fliplr(s_values)], ...
    [(mu_enc_2 + std_enc_2)*100, fliplr((mu_enc_2 - std_enc_2)*100)], ...
    color_enc, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

fill([s_values, fliplr(s_values)], ...
    [(mu_org_2 + std_org_2)*100, fliplr((mu_org_2 - std_org_2)*100)], ...
    color_org, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');

h1 = plot(s_values, mu_enc_0*100, '-o', 'LineWidth', 2.5, ...
    'Color', color_enc, 'MarkerFaceColor', color_enc, 'MarkerSize', 8);

h2 = plot(s_values, mu_org_0*100, '-o', 'LineWidth', 2.5, ...
    'Color', color_org, 'MarkerFaceColor', color_org, 'MarkerSize', 8);

h3 = plot(s_values, mu_enc_2*100, '--s', 'LineWidth', 2.5, ...
    'Color', color_enc, 'MarkerFaceColor', 'w', 'MarkerSize', 8);

h4 = plot(s_values, mu_org_2*100, '--s', 'LineWidth', 2.5, ...
    'Color', color_org, 'MarkerFaceColor', 'w', 'MarkerSize', 8);


ylim([20 100])
xlim([s_values(1) - 10, s_values(end) + 10]) 
set(gca, 'XTick', s_values)
legend([h1, h3, h2, h4], ...
    {'Latent trajectory (2-back)', 'Latent trajectory (0-back)', 'fMRI signal (2-back)', 'fMRI signal (0-back)'}, ...
    'Location', 'best', 'FontSize', 25, 'NumColumns', 2);
ax = gca;
ax.FontSize = 26;
xlabel('Subjects', 'FontSize', 30);
ylabel('Accuracy (%)', 'FontSize', 30);
grid on;
box on;
xlim([min(s_values)-2, max(s_values)+2]);

hold off;

