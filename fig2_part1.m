%% generate distribution of theta in Fig.2a
addpath('matlab_code')
load("theta\res_subject_1000.mat")
load("theta\res_phase.mat")
load("theta\res_time.mat")
time_len = 9;
start_time = 1;

the1 = squeeze(mean(abs(the), 2));
the1 = squeeze(the1(start_time:start_time+time_len, :, :));
the1 = permute(the1, [2, 1, 3]);
[t1, b1, f1] = size(the1);
data1 = reshape(the1, [t1, b1*f1]); 
data1 = data1(:);

the2 = squeeze(mean(abs(the_time), 2));
the2 = squeeze(the2(start_time:start_time+time_len, :, :));
the2 = permute(the2, [2, 1, 3]);
[t2, b2, f2] = size(the2);
data2 = reshape(the2, [t2, b2*f2]); 
data2 = data2(:);

the3 = squeeze(mean(abs(the_phase), 2));
the3 = squeeze(the3(start_time:start_time+time_len, :, :));
the3 = permute(the3, [2, 1, 3]);
[t3,b3,f3] = size(the3);
data3 = reshape(the3, [t3, b3*f3]);
data3 = data3(:);

color1 = [0.017, 0.443, 0.690];
color3 = [0.843, 0.098, 0.110];
color2 = [0.4, 0.4, 0.4];   

ax1 = axes;

h1 = histogram(ax1, data1, 'BinWidth',0.01,...
    'Normalization','pdf',...
    'FaceColor',color1,'FaceAlpha',0.5,'EdgeColor','none');

hold(ax1,'on')

h3 = histogram(ax1, data3, 'BinWidth',0.01,...
    'Normalization','pdf',...
    'FaceColor',color3,'FaceAlpha',0.5,'EdgeColor','none');

set(ax1,'XColor',color1,'YColor','k')
xlabel(ax1,'Angular frequency (rad/s)','Color','k','FontSize',39)
ylabel(ax1,'Probability Density (%)','FontSize',39)
ax1.FontSize =30;
grid(ax1,'on')
box(ax1,'on')

ax2 = axes('Position',ax1.Position,...
    'XAxisLocation','top',...
    'YAxisLocation','right',...
    'Color','none');

hold(ax2,'on')

h2 = histogram(ax2,data2,'BinWidth',0.01,...
    'Normalization','pdf',...
    'FaceColor',color2,'FaceAlpha',0.9,'EdgeColor','none');

set(ax2,'XColor',color2,'YColor','none', 'Fontsize', 30)

max_y = max([h1.Values h2.Values h3.Values]);
upper_lim = max_y*1.1;

ylim(ax1,[0 upper_lim])
ylim(ax2,[0 upper_lim])

legend(ax1,[h1 h3 h2],...
    {'Time embedding data','Phase random surrogate','Time shuffled surrogate'},...
    'Location','northeast','FontSize',30)

linkprop([ax1 ax2],'Position')

%% Kruskal-Wallis test for theta across subjects in Fig.2c
function [p_values, h_values, effect_sizes] = test_distribution_across_subjects(data)
    [n_dim, n_subjects, n_features] = size(data);
    p_values = zeros(n_dim, 1);
    h_values = zeros(n_dim, 1);
    
    for dim = 1:n_dim
        current_dim_data = squeeze(data(dim, :, :)); 
        all_data = [];
        group_labels = [];
        
        for sub = 1:n_subjects
            subject_data = current_dim_data(sub, :);
            all_data = [all_data, subject_data];
            group_labels = [group_labels, repmat(sub, 1, n_features)];
        end

        [p, tbl] = kruskalwallis(all_data, group_labels, 'off');
        
        chi_sq = tbl{2,5};
        n_total = length(all_data);
        epsilon_sq = chi_sq / (n_total - 1);
        
        p_values(dim) = p;
        h_values(dim) = p < 0.05;
        effect_sizes(dim) = epsilon_sq;
        
        fprintf('Dimension %d: p = %.6f, significant = %d\n', dim, p, epsilon_sq);
    end
end

clc
time_len = 9;
start_time = 1;
sub_num = 10:100:950;
n_num = length(sub_num);

p_sub = zeros(n_num, 426);
effect_sub = zeros(n_num, 426);
within_var_sub = zeros(n_num, 426);
between_var_sub = zeros(n_num, 426);

for sub_idx = 1:n_num
    sub = sub_num(sub_idx);
    the = the_700(start_time:start_time+time_len, 1:sub,:,:);
    the_flat = reshape(permute(the, [3, 2, 1, 4]), [426, size(the, 2), size(the, 1)*size(the, 4)]);
    [p_vals, h_vals, eff_vals] = test_distribution_across_subjects(the_flat);
    
    p_sub(sub_idx, :) = p_vals;
    effect_sub(sub_idx, :) = eff_vals;
end
sub_num = 10:100:950;
yyaxis left
mean_p = mean(p_sub, 2);
std_p = std(p_sub, 0, 2);
upper_bound = min(mean_p + std_p, 1); 
err_upper = upper_bound - mean_p;
err_lower = mean_p - max(mean_p - std_p, 0);

blue = [0.017, 0.443, 0.690];
red  = [0.843, 0.098, 0.110];

yyaxis left
errorbar(sub_num, mean_p, err_lower, err_upper, 'o-', ...
    'LineWidth', 3, 'MarkerSize', 8, ...
    'Color', blue, ...
    'MarkerFaceColor', blue);
ylabel('P-value', 'FontSize', 22);
ylim([0, 1.1]);
set(gca, 'YColor', blue);

yyaxis right
errorbar(sub_num, mean(effect_sub, 2), std(effect_sub, 0, 2), 'o-', ...
    'LineWidth', 3, 'MarkerSize', 8, ...
    'Color', red, ...
    'MarkerFaceColor', red);
ylabel('Effect Size', 'FontSize', 21);
ylim([0, 0.1]);
set(gca, 'YColor', red);

legend({'P-value', 'Effect Size (ε²)'}, 'Location', 'east', 'FontSize', 30);
xlabel('Subjects', 'FontSize', 21);
ax = gca;
ax.FontSize=30;
grid on;
xlim([0, 920]);

%% generate CDF of theta in Fig.2b
time_len = 9;
start_time = 1;
nums = [50, 300, 700]; % make sure that compared epochs exsit in work space  
                       % and name as 'the_700', specifically for epoch-700 
figure
hold on;

legend_str = {};

base_color = [0.017, 0.443, 0.690];
bg_color = [1, 1, 1];
alpha_values = linspace(0.3, 1.0, length(nums)); 

for i = 1:length(nums)
    var_name = sprintf('the_%d', nums(i));
    if exist(var_name, 'var')
        temp = eval(var_name);
        
        the = squeeze(mean(abs(temp), 2));
        
        the = squeeze(the(start_time:start_time+time_len, :, :));
        the = permute(the, [2, 1, 3]);
        [t, b, f] = size(the);
        the_unfold = reshape(the, [t, b*f]);
        data_vec = the_unfold(:); 
        
        [f_cdf, x_cdf] = ecdf(data_vec);
        
        alpha = alpha_values(i);
        mixed_color = alpha * base_color + (1 - alpha) * bg_color;
        
        plot(x_cdf, f_cdf, 'LineWidth', 2.5, 'Color', mixed_color);
        
        legend_str{end+1} = ['Train epoch=', num2str(nums(i))];
        
    else
        warning('variable %s does not exist, skip', var_name);
    end
end

x_limits = xlim;

plot(x_limits, [0, 1], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Uniform Ref');
legend_str{end+1} = 'Uniform distribution';

legend(legend_str, 'Location', 'southeast', 'FontSize', 30, 'Interpreter', 'tex');
xlabel('Angular frequency (rad/s)', 'FontSize', 30);
ylabel('Cumulative probability', 'FontSize', 30);

grid on;
box on;
ylim([0, 1]);
ax = gca;
ax.FontSize=30;