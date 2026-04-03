%% identify individual differences of face condition in Fig.4c and 4e
addpath('matlab_code')
[order_map, class_info, area_name, specify] = template_info();
csv_file = 'task\wm\wm.xlsx';
sub_inform = readtable(csv_file);
first_col_name = sub_inform.Properties.VariableNames{1};
load('task\wm\ids.mat')
% filter subjects record in score and fMRI
[is_match, match_idx] = ismember(id, sub_inform.(first_col_name));
filtered_sub_inform = sub_inform(match_idx(is_match), :);
% assign score to divide high-score/low-score group
test_values = filtered_sub_inform.WM_Task_2bk_Face_Acc; % Face as example
% load data in different Back-conditions
load("task\wm\face2_awake_0_lenth_10_time1_subject_1000.mat")
load("task\wm\face0_awake_0_lenth_10_time1_subject_1000.mat")
grad_out = abs(grad_in_face2_time1);
grad_in = abs(grad_in_face0_time1);
n_regions = 426;
time_cor = zeros(size(grad_in, 2), size(grad_in, 1), n_regions);
for sub = 1:size(grad_in,2)
    grad_st = squeeze(grad_out(1, sub, :, :));
    for t = 1:size(grad_in, 1)
        grad = squeeze(grad_in(t, sub, :, :));
        c = zeros(n_regions,1);
        for i = 1:n_regions
            A = grad_st(:, i);
            B = grad(:, i);
            c(i) = corr(A, B);
        end
        time_cor(sub, t, :) = c;
    end
end

%
n = 100; % subject number for each group
[n_subs, n_time, n_regions] = size(time_cor);
n_time = 10;

[sorted_values, sort_idx] = sort(test_values);

idx_bottom = sort_idx(1:n);           
idx_top = sort_idx(end-n+1:end);      

H_real = zeros(n_regions,1);
P_real = zeros(n_regions,1);

data_top = time_cor(idx_top, :, :);
data_bottom = time_cor(idx_bottom, :, :);
dis_top = zeros(n_regions, n_time*n);
dis_bottom = zeros(n_regions, n_time*n);

for r = 1:n_regions
    g1 = squeeze(data_top(:, 1:n_time, r));
    g2 = squeeze(data_bottom(:, 1:n_time, r));
    dis_top(r, :) = g1(:);
    dis_bottom(r, :) = g2(:);
    [H_real(r), P_real(r)] = kstest2(g1(:), g2(:));
end
disp(sum(H_real))

t_values = 1:1:10; 
n_t = length(t_values);

mean_top_H1 = zeros(1,n_t); sem_top_H1 = zeros(1,n_t);
mean_btm_H1 = zeros(1,n_t); sem_btm_H1 = zeros(1,n_t);
mean_top_H0 = zeros(1,n_t); sem_top_H0 = zeros(1,n_t);
mean_btm_H0 = zeros(1,n_t); sem_btm_H0 = zeros(1,n_t);
p_vals_H1   = zeros(1,n_t); p_vals_H0   = zeros(1,n_t);


for i = 1:n_t
    t = t_values(i);
    
    % --- H_real == 1 ---
    d_btm_1 = time_cor(idx_bottom, t, H_real==1); 
    d_top_1 = time_cor(idx_top,    t, H_real==1);

    d_btm_1 = d_btm_1(:);
    d_top_1 = d_top_1(:);
    
    mean_btm_H1(i) = mean(d_btm_1); sem_btm_H1(i) = std(d_btm_1)/sqrt(length(d_btm_1));
    mean_top_H1(i) = mean(d_top_1); sem_top_H1(i) = std(d_top_1)/sqrt(length(d_top_1));
    [~, p_vals_H1(i)] = ttest2(d_top_1, d_btm_1, 'Tail', 'right');
    
    % --- H_real == 0 ---
    d_btm_0 = time_cor(idx_bottom, t, H_real==0);
    d_top_0 = time_cor(idx_top,    t, H_real==0);
    
    d_btm_0 = d_btm_0(:);
    d_top_0 = d_top_0(:);

    mean_btm_H0(i) = mean(d_btm_0); sem_btm_H0(i) = std(d_btm_0)/sqrt(length(d_btm_0));
    mean_top_H0(i) = mean(d_top_0); sem_top_H0(i) = std(d_top_0)/sqrt(length(d_top_0));
    [~, p_vals_H0(i)] = ttest2(d_top_0, d_btm_0, 'Tail', 'right');
end

figure;

set(groot, 'defaultAxesFontSize', 16);       
set(groot, 'defaultTextFontSize', 16);       
set(groot, 'defaultLegendFontSize', 16);      
set(groot, 'defaultColorbarFontSize', 11);
hold on;

c_top = [0.843, 0.098, 0.110]; 
c_btm = [0.017, 0.443, 0.690];

h3 = errorbar(t_values, mean_top_H0, sem_top_H0, '--s', 'LineWidth', 2, ...
    'Color', [c_top, 0.6], 'MarkerFaceColor', 'w', 'CapSize', 0, 'DisplayName', 'Top (H=0)'); 
h4 = errorbar(t_values, mean_btm_H0, sem_btm_H0, '--o', 'LineWidth', 2, ...
    'Color', [c_btm, 0.6], 'MarkerFaceColor', 'w', 'CapSize', 0, 'DisplayName', 'Bottom (H=0)');

h1 = errorbar(t_values, mean_top_H1, sem_top_H1, '-s', 'LineWidth', 2, ...
    'Color', c_top, 'MarkerFaceColor', c_top, 'CapSize', 5, 'DisplayName', 'Top (H=1)');
h2 = errorbar(t_values, mean_btm_H1, sem_btm_H1, '-o', 'LineWidth', 2, ...
    'Color', c_btm, 'MarkerFaceColor', c_btm, 'CapSize', 5, 'DisplayName', 'Bottom (H=1)');

all_data = [mean_top_H1, mean_btm_H1, mean_top_H0, mean_btm_H0];
y_range = max(all_data) - min(all_data);
offset_dist = y_range * 0.03;


for i = 1:n_t
    p1 = p_vals_H1(i);
    if p1 < 0.05
        if p1 < 0.001, txt = '***'; elseif p1 < 0.01, txt = '**'; else, txt = '*'; end
        
        y_pos = mean_btm_H1(i) - offset_dist;
        
        text(t_values(i), y_pos, txt, 'Color', 'k', 'FontSize', 16, ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
            'VerticalAlignment', 'top'); 
    end    
    p0 = p_vals_H0(i);
    if p0 < 0.05
        if p0 < 0.001, txt = '***'; elseif p0 < 0.01, txt = '**'; else, txt = '*'; end
        
        y_pos = mean_top_H1(i) + offset_dist;
                
        text(t_values(i), y_pos, txt, 'Color', [0.6 0.6 0.6], 'FontSize', 14, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); 
    end
end
y_min_limit = min(mean_btm_H1 - sem_btm_H1) - offset_dist * 5;
y_max_limit = max([mean_top_H1+sem_top_H1, mean_top_H0+sem_top_H0]) + offset_dist * 4;
ylim([y_min_limit, y_max_limit]);
xlim([0.8, 10.2]);



legend([h1, h2, h3, h4], ...
    {'High score ($p<0.05$)', 'Low score ($p<0.05$)', 'High score ($p\geq0.05$)', 'Low score ($p\geq0.05$)'}, ...
    'Location', 'southwest', 'NumColumns', 2, 'FontSize', 24, 'NumColumns', 2,'Interpreter', 'latex');
ax = gca;
ax.FontSize = 28;


xlabel('Time', 'FontSize', 30);
ylabel('PCC', 'FontSize', 30);
grid on; box on;
hold off;

show_value = H_real;


figure('Position', [100, 100, 800, 620]);

L = 0.02; 
R = 0.02;  
top_h = 0.48;
top_y = 0.48;

bot_y = 0.2;
w = 0.28;
h = 0.28;
gap = 0.01;
ax1 = subplot('Position', [L, top_y, 1-L-R, top_h]);

[mask, label, oi_x, oi_y, order_map, area_name] = get_template();
[~, inverse_map] = sort(order_map);
grad_sort = show_value;
grad_sort = grad_sort(inverse_map);
area = grad_sort(1:360)';

tem_brainimg = zeros(size(mask));
for region = 1:360
    tem_brainimg(label == region) = area(region);
end
brainimg = tem_brainimg;

imagesc_brainimg(brainimg, mask, 0);
hold on;
scatter_boundary(oi_x, oi_y, nan);
clim([0, 1]);

load('masklabel.mat')
addpath('D:\HCP\nii')
load('HCPex_2mm_mask.mat')
label = load_nii('HCPex_2mm.nii');
label = label.img;
label_vec = label(mask==1);

[M,N,S] = size(mask);
[X,Y,Z] = meshgrid(1:N,1:M,1:S);
X=X-.5; Y=Y-.5; Z=Z-.5;

n = 256;
red   = [0.843, 0.098, 0.110];  
blue  = [0.017, 0.443, 0.690];   
white = [1,     1,     1    ];

half = n / 2;

CM = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];

s = 0.2;
CM = min(CM + s, 1);
colormap(CM)



value = grad_sort;
label_idx = 1;
i_task = 1;
transparent = 0.5;


caxis([0 label_idx]);

views = [90 -90; 90 0; 45 45];


Wavail = 1 - L - R;
x0 = L + (Wavail - (3*w + 2*gap))/2;

pos = [x0,            bot_y, w, h;
       x0 + (w+gap),  bot_y, w, h;
       x0 + 2*(w+gap), bot_y, w, h];

axs = gobjects(1,3);
for k = 1:3
    axs(k) = axes('Position', pos(k,:));
    hold(axs(k), 'on');

    for i_region = 361:426
        label_re = label;
        label_re(label_re ~= i_region) = 0;

        v = value(i_region, i_task);
        idx = min(256, max(1, round(256 * (v / label_idx))));

        p = patch(axs(k), isosurface(X, Y, Z, label_re, i_region/2));
        p.FaceColor = CM(idx,:);
        p.EdgeColor = 'none';
        p.FaceAlpha = transparent;

        isonormals(X, Y, Z, label_re, p);
    end

    axis(axs(k), 'vis3d', 'off', 'tight');
    view(axs(k), views(k,1), views(k,2));
    
    set(axs(k), 'Clipping', 'off');
end


%% identify individual differences of memory load in Fig.4b and 4d
fprintf('need concatenate 4 conditions data\n'); 
load("task\wm\compare_all.mat") % or
addpath('matlab_code')
[order_map, class_info, area_name, specify] = template_info();
csv_file = 'task\wm\wm.xlsx';
sub_inform = readtable(csv_file);
first_col_name = sub_inform.Properties.VariableNames{1};
load('task\wm\ids.mat')
[is_match, match_idx] = ismember(id, sub_inform.(first_col_name));
filtered_sub_inform = sub_inform(match_idx(is_match), :);
test_values = filtered_sub_inform.WM_Task_2bk_Acc;

n = 100; 
[block, n_subs, n_time, n_regions] = size(time_cor);
n_time = 10;

[sorted_values, sort_idx] = sort(test_values);

idx_bottom = sort_idx(1:n);
idx_top = sort_idx(end-n+1:end);

H_real = zeros(n_regions,1);
P_real = zeros(n_regions,1);

data_top = time_cor(:,idx_top, :, :);
data_bottom = time_cor(:,idx_bottom, :, :);
dis_top = zeros(n_regions, block*n_time*n);
dis_bottom = zeros(n_regions, block*n_time*n);

for r = 1:n_regions
    g1 = squeeze(data_top(:, :, 1:n_time, r));
    g2 = squeeze(data_bottom(:,:, 1:n_time, r));
    dis_top(r, :) = g1(:);
    dis_bottom(r, :) = g2(:);
    [H_real(r), P_real(r)] = kstest2(g1(:), g2(:));
end
disp(sum(H_real))

t_values = 1:1:10; 
n_t = length(t_values);

mean_top_H1 = zeros(1,n_t); sem_top_H1 = zeros(1,n_t);
mean_btm_H1 = zeros(1,n_t); sem_btm_H1 = zeros(1,n_t);
mean_top_H0 = zeros(1,n_t); sem_top_H0 = zeros(1,n_t);
mean_btm_H0 = zeros(1,n_t); sem_btm_H0 = zeros(1,n_t);
p_vals_H1   = zeros(1,n_t); p_vals_H0   = zeros(1,n_t);

for i = 1:n_t
    t = t_values(i);
        
    % --- H_real == 1 ---
    raw_btm_1 = time_cor(:, idx_bottom, t, H_real==1); 
    raw_top_1 = time_cor(:, idx_top,    t, H_real==1);
    
    d_btm_1 = raw_btm_1(:);
    d_top_1 = raw_top_1(:);
    
    mean_btm_H1(i) = mean(d_btm_1); sem_btm_H1(i) = std(d_btm_1)/sqrt(length(d_btm_1));
    mean_top_H1(i) = mean(d_top_1); sem_top_H1(i) = std(d_top_1)/sqrt(length(d_top_1));
    [~, p_vals_H1(i)] = ttest2(d_top_1, d_btm_1, 'Tail', 'left');
    
    % --- H_real == 0 ---
    raw_btm_0 = time_cor(:, idx_bottom, t, H_real==0);
    raw_top_0 = time_cor(:, idx_top,    t, H_real==0);
    
    
    d_btm_0 = raw_btm_0(:);
    d_top_0 = raw_btm_0(:);
    
    mean_btm_H0(i) = mean(d_btm_0); sem_btm_H0(i) = std(d_btm_0)/sqrt(length(d_btm_0));
    mean_top_H0(i) = mean(d_top_0); sem_top_H0(i) = std(d_top_0)/sqrt(length(d_top_0));
    [~, p_vals_H0(i)] = ttest2(d_top_0, d_btm_0, 'Tail', 'left');
end

figure;

set(groot, 'defaultAxesFontSize', 16);       
set(groot, 'defaultTextFontSize', 16);       
set(groot, 'defaultLegendFontSize', 16);      
set(groot, 'defaultColorbarFontSize', 11);
hold on;

c_top = [0.843, 0.098, 0.110]; 
c_btm = [0.017, 0.443, 0.690];

h3 = errorbar(t_values, mean_top_H0, sem_top_H0, '--s', 'LineWidth', 2, ...
    'Color', [c_top, 0.6], 'MarkerFaceColor', 'w', 'CapSize', 0, 'DisplayName', 'High score (H=0)'); 
h4 = errorbar(t_values, mean_btm_H0, sem_btm_H0, '--o', 'LineWidth', 2, ...
    'Color', [c_btm, 0.6], 'MarkerFaceColor', 'w', 'CapSize', 0, 'DisplayName', 'Low score (H=0)');

h1 = errorbar(t_values, mean_top_H1, sem_top_H1, '-s', 'LineWidth', 2, ...
    'Color', c_top, 'MarkerFaceColor', c_top, 'CapSize', 5, 'DisplayName', 'High score (H=1)');
h2 = errorbar(t_values, mean_btm_H1, sem_btm_H1, '-o', 'LineWidth',2, ...
    'Color', c_btm, 'MarkerFaceColor', c_btm, 'CapSize', 5, 'DisplayName', 'Low score (H=1)');

all_data = [mean_top_H1, mean_btm_H1, mean_top_H0, mean_btm_H0];
y_range = max(all_data) - min(all_data);
offset_dist = y_range * 0.04;

for i = 1:n_t
    % H=1
    p1 = p_vals_H1(i);
    if p1 < 0.05
        if p1 < 0.001, txt = '***'; elseif p1 < 0.01, txt = '**'; else, txt = '*'; end
        y_pos = mean_btm_H1(i) - sem_btm_H1(i) - offset_dist;
        text(t_values(i), y_pos, txt, 'Color', 'k', 'FontSize', 16, ...
            'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'VerticalAlignment', 'top');
    end
    % H=0
    p0 = p_vals_H0(i);
    if p0 < 0.05
        if p0 < 0.001, txt = '***'; elseif p0 < 0.01, txt = '**'; else, txt = '*'; end
        y_pos = mean_btm_H0(i) + sem_btm_H0(i) + offset_dist;
        text(t_values(i), y_pos, txt, 'Color', [0.6 0.6 0.6], 'FontSize', 14, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); 
    end
end

y_min_limit = min(mean_btm_H1 - sem_btm_H1) - offset_dist * 5;
y_max_limit = max([mean_top_H1+sem_top_H1, mean_top_H0+sem_top_H0]) + offset_dist * 4;
ylim([y_min_limit, y_max_limit]);
xlim([0.8, 10.2]);

legend([h1, h2, h3, h4], ...
    {'High score ($p<0.05$)', 'Low score ($p<0.05$)', 'High score ($p\geq0.05$)', 'Low score ($p\geq0.05$)'}, ...
    'Location', 'southwest', 'NumColumns', 2, 'FontSize', 24, 'NumColumns', 2,'Interpreter', 'latex');
ax = gca;
ax.FontSize = 28;


xlabel('Time', 'FontSize', 30);
ylabel('PCC', 'FontSize', 30);
grid on; box on;
hold off;

figure('Position', [100, 100, 800, 620]);
show_value = H_real;
L = 0.02; 
R = 0.02;  
top_h = 0.48;
top_y = 0.48;

bot_y = 0.2;
w = 0.28;
h = 0.28;
gap = 0.01;
ax1 = subplot('Position', [L, top_y, 1-L-R, top_h]);

[mask, label, oi_x, oi_y, order_map, area_name] = get_template();
[~, inverse_map] = sort(order_map);
grad_sort = show_value;
grad_sort = grad_sort(inverse_map);
area = grad_sort(1:360)';

tem_brainimg = zeros(size(mask));
for region = 1:360
    tem_brainimg(label == region) = area(region);
end
brainimg = tem_brainimg;

imagesc_brainimg(brainimg, mask, 0);
hold on;
scatter_boundary(oi_x, oi_y, nan);
clim([0, 1]);

load('masklabel.mat')
addpath('D:\HCP\nii')
load('HCPex_2mm_mask.mat')
label = load_nii('HCPex_2mm.nii');
label = label.img;
label_vec = label(mask==1);

[M,N,S] = size(mask);
[X,Y,Z] = meshgrid(1:N,1:M,1:S);
X=X-.5; Y=Y-.5; Z=Z-.5;

n = 256;
red   = [0.843, 0.098, 0.110];  
blue  = [0.017, 0.443, 0.690];   
white = [1,     1,     1    ];

half = n / 2;

CM = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];

s = 0.2;
CM = min(CM + s, 1);
colormap(CM)



value = grad_sort;
label_idx = 1;
i_task = 1;
transparent = 0.5;


caxis([0 label_idx]);

views = [90 -90; 90 0; 45 45];


Wavail = 1 - L - R;
x0 = L + (Wavail - (3*w + 2*gap))/2;

pos = [x0,            bot_y, w, h;
       x0 + (w+gap),  bot_y, w, h;
       x0 + 2*(w+gap), bot_y, w, h];

axs = gobjects(1,3);
for k = 1:3
    axs(k) = axes('Position', pos(k,:));
    hold(axs(k), 'on');

    for i_region = 361:426
        label_re = label;
        label_re(label_re ~= i_region) = 0;

        v = value(i_region, i_task);
        idx = min(256, max(1, round(256 * (v / label_idx))));

        p = patch(axs(k), isosurface(X, Y, Z, label_re, i_region/2));
        p.FaceColor = CM(idx,:);
        p.EdgeColor = 'none';
        p.FaceAlpha = transparent;

        isonormals(X, Y, Z, label_re, p);
    end

    axis(axs(k), 'vis3d', 'off', 'tight');
    view(axs(k), views(k,1), views(k,2));
    
    set(axs(k), 'Clipping', 'off');
end