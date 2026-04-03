%% generate five movement of motor task in Fig.3b and corresponding Extended data
addpath('matlab_code')
[order_map, class_label, area_name, specify] = template_info();
load("task\mot\t_awake_0_lenth_5_time2_subject_1000.mat") % load specific movement
grad_out = abs(grad_out(:, : ,:, :));                 % fixtion condition, need load tongue first    
grad_in = abs(grad_in_t_time2(:, :, :, :));           % assign different movement
n_regions = 426;
time_cor = zeros(size(grad_in, 1), n_regions);
grad_st = squeeze(mean(grad_out(1, :, :, :),2));
for t = 1:size(grad_in, 1)
    grad = squeeze(mean(grad_in(t, :, :, :),2));
    c = zeros(n_regions,1);
    for i = 1:n_regions
        A = grad_st(:, i);
        B = grad(:, i);
        c(i) = corr(A, B);
    end
    time_cor(t, :) = c;
end
time_cor_mean = squeeze(mean(time_cor,1));
disp('finish')
T = 5;

scatter_size = 30;
show_value = 1-time_cor;
figure('Position', [100, 100, 800, 400]);  
subplot('Position', [0, 0, 1, 1]);  
[~, inverse_map] = sort(order_map);
grad_sort = show_value(T,:);


grad_sort = grad_sort(inverse_map);
area = grad_sort(1:360)';

min_val = min(grad_sort(:));
max_val = max(grad_sort(:));

tem_brainimg = zeros(size(mask));
for region = 1:360
    tem_brainimg(label == region) = area(region);
end
brainimg = tem_brainimg;

imagesc_brainimg(brainimg, mask, 0); 
hold on; 
scatter_boundary(oi_x, oi_y, nan);
n = 256;
red   = [0.843, 0.098, 0.110];  
blue  = [0.017, 0.443, 0.690];   
white = [1,     1,     1    ];

half = n / 2;

CM = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];
colormap(CM)
cmin = 0;
cmax = max_val;  
%% compare five movement in Fig.3c
data_dir = "task\mot\";
files    = {'lf', 'rf', 'rh', 'lh', 'tongue'};
n_files  = length(files);
T        = 5;
t_top    = 10;   
t_min    = 0.8;   

task_data = zeros(n_files, 5, 426);
for i = 1:n_files
    fpath = fullfile(data_dir, [files{i} '.mat']);
    tmp   = load(fpath, 'time_cor');
    task_data(i, :, :) = tmp.time_cor;
    fprintf('Loaded %s\n', files{i});
end

low_color = [0.85, 0.85, 0.85];
high_colors = [
    0.89 0.47 0.76;  
    0.106, 0.471, 0.216;   
    0.851, 0.373, 0.008;  
    0.318, 0.118, 0.510;   
    0.843, 0.098, 0.110; 
];


[mask, label, oi_x, oi_y, order_map, area_name] = get_template();
[~, inverse_map] = sort(order_map);

task_area    = zeros(n_files, 360);
task_top_idx = false(n_files, 360);

for i = 1:n_files
    show_value = squeeze(1 - task_data(i, :, :));
    grad_sort  = show_value(T, :);
    grad_sort  = grad_sort(inverse_map);
    area       = grad_sort(1:360)';
    task_area(i, :) = area;

    [~, top_idx] = sort(area, 'descend');
    task_top_idx(i, top_idx(1:t_top)) = true;
end

task_t = zeros(n_files, 360);
for i = 1:n_files
    top_regions = find(task_top_idx(i, :));
    vals        = task_area(i, top_regions);
    v_min       = min(vals);
    v_max       = max(vals);
    t_raw       = (vals - v_min) / (v_max - v_min + 1e-8);
    task_t(i, top_regions) = t_min + (1 - t_min) * t_raw;   
end

[H, W]    = size(mask);
brain_rgb = ones(H, W, 3);

bg_color = [0.85, 0.85, 0.85];
for region = 1:360
    [rows, cols] = find(label == region);
    if isempty(rows), continue; end
    for k = 1:length(rows)
        brain_rgb(rows(k), cols(k), :) = bg_color;
    end
end

for region = 1:360
    [rows, cols] = find(label == region);
    if isempty(rows), continue; end

    selected = find(task_top_idx(:, region));
    n_sel    = length(selected);

    if n_sel == 0
        continue;

    elseif n_sel == 1
        task_idx = selected(1);
        t        = t_min;
        color    = (1 - t) * low_color + t * high_colors(task_idx, :);

        for k = 1:length(rows)
            brain_rgb(rows(k), cols(k), :) = color;
        end

    else
        pts    = [cols, rows];
        pts_c  = pts - mean(pts, 1);
        [V, ~] = eig(pts_c' * pts_c);
        axis1  = V(:, end);

        proj = pts_c * axis1;
        [~, sort_order] = sort(proj);

        n_pix   = length(rows);
        seg_len = floor(n_pix / n_sel);

        for s = 1:n_sel
            task_idx = selected(s);
            t        = t_min;
            color    = (1 - t) * low_color + t * high_colors(task_idx, :);

            if s < n_sel
                seg_idx = sort_order(((s-1)*seg_len + 1) : (s*seg_len));
            else
                seg_idx = sort_order(((s-1)*seg_len + 1) : end);
            end

            for k = 1:length(seg_idx)
                brain_rgb(rows(seg_idx(k)), cols(seg_idx(k)), :) = color;
            end
        end
    end
end

for c = 1:3
    ch = brain_rgb(:, :, c);
    ch(~mask) = 1;
    brain_rgb(:, :, c) = ch;
end

figure('Position', [100, 100, 900, 450]);
subplot('Position', [0, 0, 0.88, 1]);

h = image(brain_rgb);
set(h, 'AlphaData', mask == 1);
ax        = gca;
ax.YDir   = 'normal';
axis tight; axis equal;
ax.XTick  = []; ax.YTick  = [];
ax.Color  = 'none';
ax.XColor = 'none';
ax.YColor = 'none';

hold on;
scatter_boundary(oi_x, oi_y, 1.5);

labels    = {'lf', 'rf', 'rh', 'lh', 'tongue'};
legend_ax = axes('Position', [0.88, 0.15, 0.10, 0.70]);
axis off; hold on;

n_grad = 50;
for i = 1:n_files
    y_bot = (i - 1) / n_files;
    y_top =  i      / n_files;
    y_mid = (y_bot + y_top) / 2;

    for k = 1:n_grad
        t_k   = t_min + (1 - t_min) * (k - 1) / (n_grad - 1);  
        col_k = (1 - t_k) * low_color + t_k * high_colors(i, :);
        x_l   = (k - 1) / n_grad * 0.6;
        x_r   =  k       / n_grad * 0.6;
        fill([x_l x_r x_r x_l], ...
             [y_bot+0.02 y_bot+0.02 y_top-0.02 y_top-0.02], ...
             col_k, 'EdgeColor', 'none');
    end

    text(0.65, y_mid, labels{i}, ...
         'FontSize', 11, 'VerticalAlignment', 'middle', ...
         'Color', high_colors(i, :), 'FontWeight', 'bold');
end
xlim([0 1.2]); ylim([0 1]);

