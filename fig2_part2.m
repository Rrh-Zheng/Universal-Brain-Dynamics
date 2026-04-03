%% correlation between dynamics, FC, SC in Fig.2e
addpath("matlab_code")
clc; clear;
load('dataset\fc.mat')
load('dataset\dti.mat')
load("rest\length_10_subject_1000.mat")
% make sure data exsit (created by run.py)
sub_num = 963;

c_joc_sc = zeros(sub_num, size(grad,1), size(grad,3));
c_joc_fc = zeros(sub_num, size(grad,1), size(grad,3));

for sub = 1:sub_num
    fc = squeeze(fc_all(sub, :,:));
    grad_sub = squeeze(grad(:,sub, :, :));
    for t = 1:size(grad_sub,1)
        grad_temp = squeeze(grad_sub(t,:,:));
        grad_cor = corr(grad_temp',grad_temp');
        for i = 1 : size(grad_sub,2)
            c_joc_sc(sub, t, i) = corr(grad_cor(i,:)', dti(i,:)');
            c_joc_fc(sub, t, i) = corr(grad_cor(i,:)', fc(i,:)');
        end
    end
end

c_joc_sc_mean = squeeze(mean(c_joc_sc, 3));   
c_joc_fc_mean = squeeze(mean(c_joc_fc, 3));

mean_sc = squeeze(mean(c_joc_sc_mean, 1));
mean_fc = squeeze(mean(c_joc_fc_mean, 1));

std_sc_sub = squeeze(std(c_joc_sc_mean, 0, 1));
std_fc_sub = squeeze(std(c_joc_fc_mean, 0, 1));

mean_sc    = mean_sc(:)';
mean_fc    = mean_fc(:)';
std_sc_sub = std_sc_sub(:)';
std_fc_sub = std_fc_sub(:)';

t = 1:length(mean_sc);

color1 = [0.017, 0.443, 0.690];
color2 = [0.843, 0.098, 0.110];

figure;
hold on

% --- Plot SC Data (Color 1) ---
plot(t, mean_sc, 'Color', color1, 'LineWidth', 2);
fill([t, fliplr(t)], [mean_sc - std_sc_sub, fliplr(mean_sc + std_sc_sub)], ...
     color1, 'FaceAlpha', 0.35, 'EdgeColor', 'none');

% --- Plot FC Data (Color 2) ---
plot(t, mean_fc, 'Color', color2, 'LineWidth', 2);
fill([t, fliplr(t)], [mean_fc - std_fc_sub, fliplr(mean_fc + std_fc_sub)], ...
     color2, 'FaceAlpha', 0.35, 'EdgeColor', 'none');

xlabel('Time', 'FontSize', 22);
ylabel('PCC', 'FontSize', 22);
xlim([1 10])

legend({'Mean correlation with SC', 'Subject standard deviation (SC)', ...
        'Mean correlation with FC', 'Subject standard deviation (FC)'}, ...
        'Location', 'best', 'FontSize', 30);
grid on;
box on;
ax = gca;
ax.FontSize = 30;
set(gcf, 'Units', 'inches');
set(gcf, 'Position', [1, 1, 16, 9]);
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10, 6]);
set(gcf, 'PaperPosition', [0, 0, 10, 6]);
%% Generate pairwise correlation of brain dynamics in Fig.2d and Fig.2f
n = 256;
red   = [0.843, 0.098, 0.110];
blue  = [0.017, 0.443, 0.690];
white = [1,     1,     1    ];
half = n / 2;
cmap = [linspace(blue(1),  white(1), half)', linspace(blue(2),  white(2), half)', linspace(blue(3),  white(3), half)';
        linspace(white(1),  red(1),  half)', linspace(white(2),  red(2),  half)', linspace(white(3),  red(3),  half)'];

figure('Position', [100, 100, 800, 700]);
jc = squeeze(abs(grad(1,:,:,:)));
show_value = squeeze(mean(jc,1));
show_value = corr(show_value',show_value');
triangle = show_value;  
h = imagesc(triangle);
set(h, 'AlphaData', ~isnan(triangle));
rate = 20;
clim([min(show_value(:))/rate, max(show_value(:))/rate])
colormap(cmap)
axis square;
axis off; 

figure('Position', [100, 100, 800, 700]);
jc = squeeze(abs(grad(10,:,:,:)));
show_value = squeeze(mean(jc,1));
show_value = corr(show_value',show_value');
triangle = show_value;  
h = imagesc(triangle);
set(h, 'AlphaData', ~isnan(triangle));
rate = 4;
clim([min(show_value(:))/rate, max(show_value(:))/rate])
colormap(cmap)
axis square;
axis off; 
