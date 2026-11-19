
function s2n(varargin)
    % Parse optional input arguments
    p = inputParser;
    addParameter(p, 'repo_root', pwd, @ischar);
    addParameter(p, 'reg_csv', [], @(x) ischar(x) || isstring(x));
    addParameter(p, 'group_csv', [], @(x) ischar(x) || isstring(x));
    addParameter(p, 'out_dir', [], @(x) ischar(x) || isstring(x));
    addParameter(p, 'use_mae_per_100c', false, @islogical);
    addParameter(p, 'figure_width_mm', 183, @isnumeric);
    addParameter(p, 'figure_height_mm', 82, @isnumeric);
    addParameter(p, 'dpi_export', 300, @isnumeric);
    
    parse(p, varargin{:});
    
    repo_root = p.Results.repo_root;
    use_mae_per_100c = p.Results.use_mae_per_100c;
    fig_width_mm = p.Results.figure_width_mm;
    fig_height_mm = p.Results.figure_height_mm;
    dpi_export = p.Results.dpi_export;
    
    %% ========== Configuration: CSV file paths ==========
    if isempty(p.Results.reg_csv)
        reg_csv = fullfile(repo_root, 'checkpoints_s2n_v1', 'eval', ...
            'plot_data', 'test_regression_points.csv');
    else
        reg_csv = p.Results.reg_csv;
    end
    
    if isempty(p.Results.group_csv)
        group_csv = fullfile(repo_root, 'checkpoints_s2n_v1', 'eval', ...
            'plot_data', 'test_group_error_summary.csv');
    else
        group_csv = p.Results.group_csv;
    end
    
    if isempty(p.Results.out_dir)
        out_dir = fullfile(repo_root, 'analysis_report', 'ncomms_s2n');
    else
        out_dir = p.Results.out_dir;
    end
    
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    
    %% ========== Nature-style parameters ==========
    % Font configuration
    font_config = struct();
    font_config.name = 'Arial';  % Helvetica acceptable alternative
    font_config.size_panel = 8.0;    % Panel labels (a, b, c)
    font_config.size_label = 7.0;    % Axis labels, titles
    font_config.size_axis = 6.2;     % Axis tick labels
    font_config.size_annot = 6.5;    % Annotations (R², MAE values)
    font_config.size_tick_label = 5.5;   % Bar value labels
    
    % Line and marker styles
    line_width = 0.8;
    marker_size = 4;
    
    % Unit conversion (mm to cm for MATLAB)
    mm_to_cm = 0.1;
    
    % Figure sizing
    fig_width_cm = fig_width_mm * mm_to_cm;
    fig_height_cm = fig_height_mm * mm_to_cm;
    
    %% ========== Data loading and validation ==========
    try
        reg_tbl = readtable(reg_csv);
        group_tbl = readtable(group_csv);
    catch ME
        error('Failed to load CSV files: %s', ME.message);
    end
    
    % Extract data
    x = reg_tbl.true_count;
    y = reg_tbl.pred_count;
    
    % Quality control
    valid_idx = isfinite(x) & isfinite(y);
    x = x(valid_idx);
    y = y(valid_idx);
    
    % Compute metrics
    mae = mean(abs(y - x), 'omitnan');
    r2 = compute_r2(x, y);
    
    fprintf('Data loaded: %d valid points\n', length(x));
    fprintf('R² = %.4f, MAE = %.4f\n', r2, mae);
    
    %% ========== Color palette (colorblind-friendly) ==========
    % Based on Wong et al. (2011) and Okabe-Ito palette
    colors = struct();
    colors.black = [0, 0, 0] / 255;
    colors.orange = [230, 159, 0] / 255;
    colors.sky_blue = [86, 180, 233] / 255;
    colors.green = [0, 158, 115] / 255;
    colors.blue = [0, 114, 178] / 255;
    colors.vermillion = [213, 94, 0] / 255;
    colors.purple = [204, 121, 167] / 255;
    colors.gray = [115, 115, 115] / 255;
    
    bar_colors = [
        colors.orange;
        colors.blue;
        colors.green;
        colors.vermillion;
        colors.purple
    ];
    
    %% ========== Figure creation ==========
    fig = figure('Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [1.0, 1.0, fig_width_cm, fig_height_cm], ...
        'Renderer', 'painters', ...  % Vector rendering
        'PaperPositionMode', 'auto', ...
        'NumberTitle', 'off', ...
        'Name', 'S2N_NatureComm_Figure');
    
    % Create tiled layout
    tl = tiledlayout(fig, 1, 2, ...
        'Padding', 'compact', ...
        'TileSpacing', 'compact', ...
        'Units', 'centimeters');
    
    %% ========== PANEL A: Density regression ==========
    ax1 = nexttile(tl, 1);
    hold(ax1, 'on');
    axis(ax1, 'equal');
    
    % Determine data range
    max_val = ceil(max([x; y]) * 1.05);
    
    % Create 2D histogram
    bin_edges = linspace(0, max_val, 70);
    h = histogram2(ax1, x, y, bin_edges, bin_edges, ...
        'DisplayStyle', 'tile', ...
        'ShowEmptyBins', 'off', ...
        'Normalization', 'count', ...
        'EdgeColor', 'none');
    
    % Apply logarithmic colorscale
    ax1.ColorScale = 'log';
    colormap(ax1, build_nature_colormap(256));
    
    % Diagonal reference line
    plot(ax1, [0, max_val], [0, max_val], '--', ...
        'Color', colors.black, ...
        'LineWidth', line_width, ...
        'HandleVisibility', 'off');
    
    % Axis limits and formatting
    xlim(ax1, [0, max_val]);
    ylim(ax1, [0, max_val]);
    
    set(ax1, 'Box', 'off', ...
        'LineWidth', line_width, ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_axis, ...
        'TickDir', 'out', ...
        'TickLength', [0.012, 0.012], ...
        'Layer', 'top', ...
        'XGrid', 'off', ...
        'YGrid', 'off');
    
    % Axis labels
    xlabel(ax1, 'True SU count', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    ylabel(ax1, 'Predicted SU count', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    title(ax1, 'S2N test-set regression', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    
    % Statistics annotation
    txt = sprintf('R² = %.3f\nMAE = %.3f', r2, mae);
    text(ax1, 0.05, 0.93, txt, ...
        'Units', 'normalized', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_annot, ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', ...
        'Color', colors.black, ...
        'BackgroundColor', 'none', ...
        'EdgeColor', 'none');
    
    % Colorbar
    cb = colorbar(ax1, 'Location', 'EastOutside');
    cb.Label.String = 'Count';
    cb.Label.FontName = font_config.name;
    cb.Label.FontSize = font_config.size_axis;
    cb.FontName = font_config.name;
    cb.FontSize = font_config.size_tick_label;
    cb.Box = 'off';
    
    % Panel label
    text(ax1, -0.16, 1.05, 'a', ...
        'Units', 'normalized', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_panel, ...
        'FontWeight', 'bold', ...
        'Color', colors.black);
    
    %% ========== PANEL B: Group-wise error ==========
    ax2 = nexttile(tl, 2);
    hold(ax2, 'on');
    
    % Group definitions
    group_order = {'carbonyl', 'aromatic', 'unsaturated', 'aliphatic', 'hetero_terminal'};
    group_labels = {'Carbonyl', 'Aromatic', 'Unsaturated', 'Aliphatic', 'Hetero-terminal'};
    
    % Metric selection
    if use_mae_per_100c
        metric_name = 'mae_per_100C';
        y_label = 'MAE per 100 C';
    else
        metric_name = 'mae';
        y_label = 'MAE (count)';
    end
    
    % Extract group values
    vals = zeros(numel(group_order), 1);
    for i = 1:numel(group_order)
        idx = strcmp(group_tbl.group, group_order{i});
        if any(idx)
            vals(i) = group_tbl.(metric_name)(find(idx, 1, 'first'));
        else
            vals(i) = NaN;
        end
    end
    
    % Bar plot with custom colors
    b = bar(ax2, vals, 0.68, ...
        'FaceColor', 'flat', ...
        'EdgeColor', colors.black, ...
        'LineWidth', 0.6);
    b.CData = bar_colors;
    
    % Value labels on top of bars
    for i = 1:numel(vals)
        if isfinite(vals(i))
            text(ax2, i, vals(i) + max(vals, [], 'omitnan') * 0.03, ...
                sprintf('%.2f', vals(i)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontName', font_config.name, ...
                'FontSize', font_config.size_tick_label, ...
                'Color', colors.black);
        end
    end
    
    % Axis formatting
    set(ax2, 'Box', 'off', ...
        'LineWidth', line_width, ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_axis, ...
        'TickDir', 'out', ...
        'TickLength', [0.012, 0.012], ...
        'Layer', 'top', ...
        'XTick', 1:numel(group_labels), ...
        'XTickLabel', group_labels, ...
        'XTickLabelRotation', 28, ...
        'YGrid', 'off');
    
    ylim(ax2, [0, max(vals, [], 'omitnan') * 1.15]);
    
    % Axis labels
    xlabel(ax2, 'SU group', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    ylabel(ax2, y_label, ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    title(ax2, 'Group-wise prediction error', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_label, ...
        'FontWeight', 'normal');
    
    % Panel label
    text(ax2, -0.16, 1.05, 'b', ...
        'Units', 'normalized', ...
        'FontName', font_config.name, ...
        'FontSize', font_config.size_panel, ...
        'FontWeight', 'bold', ...
        'Color', colors.black);
    
    %% ========== Export to multiple formats ==========
    % Ensure text is editable (TrueType 42 for PDF compatibility)
    set(fig, 'PaperPositionMode', 'auto');
    
    % Vector PDF (primary format for Nature)
    pdf_path = fullfile(out_dir, 'S2N_Nature_style.pdf');
    exportgraphics(fig, pdf_path, 'ContentType', 'vector', 'Resolution', dpi_export);
    fprintf('✓ Saved: %s\n', pdf_path);
    
    % High-resolution PNG for presentations
    png_path = fullfile(out_dir, 'S2N_Nature_style.png');
    exportgraphics(fig, png_path, 'Resolution', dpi_export);
    fprintf('✓ Saved: %s\n', png_path);
    
    % Vector EPS (alternative format)
    eps_path = fullfile(out_dir, 'S2N_Nature_style.eps');
    print(fig, eps_path, '-depsc2', '-vector');
    fprintf('✓ Saved: %s\n', eps_path);
    
    % Output summary
    fprintf('\n================== Export Summary ==================\n');
    fprintf('Output directory: %s\n', out_dir);
    fprintf('Figure dimensions: %.1f × %.1f mm\n', fig_width_mm, fig_height_mm);
    fprintf('Export DPI: %d\n', dpi_export);
    fprintf('Font: %s (Nature standard)\n', font_config.name);
    fprintf('===================================================\n');
    
end

%% ========== HELPER FUNCTIONS ==========

function r2 = compute_r2(x, y)
    %COMPUTE_R2 Calculate R² coefficient of determination
    x = x(:);
    y = y(:);
    
    % Remove NaN values
    mask = isfinite(x) & isfinite(y);
    x = x(mask);
    y = y(mask);
    
    % Calculate R²
    ss_res = sum((x - y).^2);
    ss_tot = sum((x - mean(x)).^2);
    r2 = 1 - ss_res / ss_tot;
    
    % Ensure R² is in valid range
    r2 = max(min(r2, 1), 0);
end

function cmap = build_nature_colormap(n)
    %BUILD_NATURE_COLORMAP Create a blue perceptually uniform colormap
    %   Similar to matplotlib's 'Blues' colormap, optimized for publication
    
    if nargin < 1
        n = 256;
    end
    
    % Anchor points for smooth interpolation
    anchor = [
        1.00, 1.00, 1.00;    % white
        0.88, 0.94, 0.98;    % very light blue
        0.64, 0.80, 0.92;    % light blue
        0.26, 0.54, 0.78;    % medium blue
        0.00, 0.28, 0.52     % dark blue
    ];
    
    % Interpolate to desired resolution
    x = linspace(0, 1, size(anchor, 1));
    xq = linspace(0, 1, n);
    cmap = interp1(x, anchor, xq, 'pchip');
    
    % Ensure valid RGB values
    cmap = min(max(cmap, 0), 1);
end