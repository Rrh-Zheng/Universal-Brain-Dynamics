%% generate presentation and response of langurage task in Fig.3f and Fig.3g
addpath('matlab_code')
[order_map, class_label, area_name, specify] = template_info();
load("task\lan\present_awake_0_lenth_10_time2_subject_1000.mat")
load("task\lan\response_awake_0_lenth_10_time2_subject_1000.mat")
grad_out = abs(grad_out(1, : ,:, :));  % inter-block period, need load present first
grad_in = abs(grad_in_present_time2(:, :, :, :));
n_regions = 426;
time_cor = zeros(size(grad_in, 1), n_regions);
grad_st =squeeze(mean(grad_out(1, :, :, :),2));
for t = 1:size(grad_in, 1)
    grad = squeeze(mean(grad_in(t, :, :, :), 2));
    c = zeros(n_regions,1);
    for i = 1:n_regions
        A = grad_st(:, i);
        B = grad(:, i);
        c(i) = corr(A, B);
    end
    time_cor(t, :) = c;
end
time_cor_mean = squeeze(mean(time_cor,2));
disp('finish')

T = 10;
show_value = 1-time_cor;


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
grad_sort = show_value(T,:);

grad_sort = grad_sort(inverse_map);
area = grad_sort(1:360)';

min_val = min(grad_sort(:));
max_val = max(grad_sort(:));
clim([min_val*2 max_val*2])

tem_brainimg = zeros(size(mask));
for region = 1:360
    tem_brainimg(label == region) = area(region);
end
brainimg = tem_brainimg;

imagesc_brainimg(brainimg, mask, 0); 
hold on; 
scatter_boundary(oi_x, oi_y, nan);

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


value = grad_sort';
label_idx = 1;
i_task = 1;
transparent = 0.5;

colormap(CM);
compare = 2;

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
        idx = min(256, max(1, round(256 * compare * (v / label_idx))));

        p = patch(axs(k), isosurface(X, Y, Z, label_re, i_region/2));
        p.FaceColor = CM(idx,:);
        p.EdgeColor = 'none';
        p.FaceAlpha = transparent;

        isonormals(X, Y, Z, label_re, p);
    end

    axis(axs(k), 'vis3d', 'off', 'tight');
    view(axs(k), views(k,1), views(k,2));

    set(axs(k), 'Clipping', 'off');
    caxis(axs(k), [min_val max_val]);
    
end
n = 256;
red   = [0.843, 0.098, 0.110];  
blue  = [0.017, 0.443, 0.690];   
white = [1,     1,     1    ];

half = n / 2;
cmap = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];
colormap(cmap)

%% compare different condition in Fig.3d and Fig.3e
load("task\lan\response_compare_all.mat") % assign benchmark
% the sequence need to adapt benchmark
var_names = { ...
    'Presentation b1', 'Presentation b2', 'Presentation b3', ...
    'Response b1', 'Response b3', ...
    'Question b1', 'Question b2', 'Question b3', ...
    'Inter-block'};
n_comparisons = 9;
color_pres = [0.017, 0.443, 0.690]; 
color_resp = [0.843, 0.098, 0.110]; 
color_ques = [0.106, 0.471, 0.216];
color_out  = [0.50, 0.50, 0.50];

my_colors = zeros(n_comparisons, 3);
% the color need to adapt benchmark
% 1. Present
my_colors(1, :) = color_pres + 0.2; 
my_colors(2, :) = color_pres;       
% my_colors(3, :) = color_pres - 0.15;

% 2. Resonse
my_colors(3, :) = color_resp + 0.2;  
my_colors(4, :) = color_resp;        
my_colors(5, :) = color_resp - 0.15; 

% 3. Question 
my_colors(6, :) = color_ques + 0.15;
my_colors(7, :) = color_ques;        
my_colors(8, :) = color_ques - 0.15; 

% 4. Out
my_colors(9, :) = color_out;

my_colors(my_colors > 1) = 1;
my_colors(my_colors < 0) = 0;

figure('Position', [100, 100, 1200, 600], 'Color', 'w');
addpath('matlab_code\Violinplot-Matlab-master');
vs = violinplot(results_matrix, var_names, 'Bandwidth', 0.01);

for i = 1:n_comparisons
    vs(i).ViolinPlot.FaceColor = my_colors(i, :);
    vs(i).ViolinPlot.EdgeColor = 'none'; 
    vs(i).ViolinPlot.FaceAlpha = 0.6;
    
    vs(i).ShowData = true; 
    vs(i).ScatterPlot.SizeData = 10;
    vs(i).ScatterPlot.MarkerFaceColor = [0.3 0.3 0.3];
    vs(i).ScatterPlot.MarkerEdgeColor = 'none'; 
    
    num_points = length(vs(i).ScatterPlot.XData);
    vs(i).ScatterPlot.XData = repmat(i, num_points, 1);
    
    vs(i).WhiskerPlot.Color = 'none'; 
    vs(i).BoxPlot.EdgeColor = 'none'; 
    vs(i).BoxPlot.FaceColor = 'none'; 
    
    vs(i).MedianPlot.Marker = 'o';
    vs(i).MedianPlot.MarkerFaceColor = 'w';
    vs(i).MedianPlot.MarkerEdgeColor = [0.2 0.2 0.2]; 
    vs(i).MedianPlot.SizeData = 30;
end

ylabel('PCC', 'FontSize', 21);
grid on;
xtickangle(30); 
ax = gca;
ax.FontSize = 21;
ylim([0.8, 1.005]);

hold on;
h = zeros(4, 1); 
h(1) = plot(nan, nan, 's', 'MarkerFaceColor', color_pres, 'MarkerEdgeColor', 'none', 'DisplayName', 'Presentation');
h(2) = plot(nan, nan, 's', 'MarkerFaceColor', color_resp, 'MarkerEdgeColor', 'none', 'DisplayName', 'Response');
h(3) = plot(nan, nan, 's', 'MarkerFaceColor', color_ques, 'MarkerEdgeColor', 'none', 'DisplayName', 'Question');
h(4) = plot(nan, nan, 's', 'MarkerFaceColor', color_out,  'MarkerEdgeColor', 'none', 'DisplayName', 'Inter-block');
legend(h, 'Location', 'southeast', 'FontSize', 21);
hold off;
