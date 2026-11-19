"""Plot S2N/G2S evaluation data with consistent per-SU count errors.

Run ``python plot/fig3/plot.py --figure_set all --out_dir OUTPUT``.
Run ``plot_predictions.py`` for prediction examples and the six-panel figure.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SU_GROUPS = {
    'carbonyl': tuple(range(0, 5)),
    'aromatic': tuple(range(5, 14)),
    'unsaturated': tuple(range(14, 19)),
    'aliphatic': tuple(range(19, 26)),
    'hetero_terminal': tuple(range(26, 33)),
}


def validate_s2n_points(points):
    """Require one finite, nonnegative count pair per molecule and SU type."""
    required = {'sample_index', 'path', 'su_index', 'true_count', 'pred_count'}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f'Missing S2N columns: {sorted(missing)}')
    if points.empty or points[list(required)].isna().any().any():
        raise ValueError('S2N data must be nonempty and contain no missing values.')
    numeric = points[['su_index', 'true_count', 'pred_count']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError('S2N indices and counts must be finite and nonnegative.')
    if not np.isin(points['su_index'], np.arange(33)).all():
        raise ValueError('S2N su_index must be an integer from 0 to 32.')
    if points['path'].astype(str).str.strip().eq('').any():
        raise ValueError('S2N source paths must not be empty.')
    if points.groupby('sample_index')['path'].nunique().ne(1).any():
        raise ValueError('Each S2N sample_index must identify one source path.')
    if points.groupby('path')['sample_index'].nunique().ne(1).any():
        raise ValueError('Each S2N source path must identify one sample_index.')
    if points.duplicated(['path', 'su_index']).any():
        raise ValueError('Duplicate S2N molecule/SU pairs are not allowed.')
    if points.groupby('path').size().ne(33).any():
        raise ValueError('Every S2N molecule must include all 33 SU types, including zeros.')


def summarize_s2n_errors(points):
    """Average absolute errors before, not after, pooling SU types."""
    validate_s2n_points(points)
    working = points.copy()
    working['absolute_count_error'] = (
        working['pred_count'] - working['true_count']
    ).abs()
    carbon_counts = working[working['su_index'] < 26].groupby('path')['true_count'].sum()
    if (carbon_counts <= 0).any():
        raise ValueError('Per-100-C errors require a positive carbon count for every molecule.')
    working['absolute_error_per_100C'] = (
        working['absolute_count_error'] * 100.0 / working['path'].map(carbon_counts)
    )
    per_su = working.groupby('su_index').agg(
        n_molecules=('path', 'nunique'),
        mae=('absolute_count_error', 'mean'),
        mae_per_100C=('absolute_error_per_100C', 'mean'),
    ).reset_index()
    if 'su_name' in working:
        if working.groupby('su_index')['su_name'].nunique().ne(1).any():
            raise ValueError('SU names must be consistent across molecules.')
        names = working.groupby('su_index')['su_name'].first()
        per_su.insert(1, 'su_name', per_su['su_index'].map(names))
    summaries = []
    for group_name, su_types in SU_GROUPS.items():
        subset = working[working['su_index'].isin(su_types)]
        summaries.append({
            'group': group_name,
            'n_su_types': len(su_types),
            'n_molecules': subset['path'].nunique(),
            'n_pairs': len(subset),
            'per_su_mae': subset['absolute_count_error'].mean(),
            'per_su_mae_per_100C': subset['absolute_error_per_100C'].mean(),
        })
    return pd.DataFrame(summaries), per_su

# ============================================================================
# Configuration: Nature Submission Standards
# ============================================================================

class NatureStyleConfig:
    """Configuration for Nature Communications figure style"""
    
    # Font settings (Nature standard: Arial or Helvetica)
    FONT_NAME = 'Arial'
    FONT_SIZE_PANEL = 8.0       # Panel labels for multi-panel figures
    FONT_SIZE_LABEL = 7.0       # Axis labels
    FONT_SIZE_AXIS = 7.0        # Numeric axis tick labels
    FONT_SIZE_AXIS_CAT = 7.0    # Categorical x-axis tick labels
    FONT_SIZE_ANNOT = 7.0       # Annotations (R^2, MAE)
    FONT_SIZE_TICK = 7.0        # Bar value labels
    
    # Line styles
    LINE_WIDTH = 0.8
    MARKER_SIZE = 4
    BAR_WIDTH = 0.58
    
    # Standalone figure dimensions (mm)
    REG_FIG_WIDTH_MM = 75       # ~7.5 cm width
    REG_FIG_HEIGHT_MM = 72      # Near-square layout for regression + colorbar
    BAR_FIG_WIDTH_MM = 75       # ~7.5 cm width
    BAR_FIG_HEIGHT_MM = 58      # Compact standalone bar chart
    ABLATION_FIG_WIDTH_MM = 75  # ~7.5 cm width
    ABLATION_FIG_HEIGHT_MM = 58 # Compact standalone line plot
    
    # Export settings
    DPI = 600
    RENDERER = 'pdf'        # Vector format for submission
    SHOW_TITLES = False     # Nature figures typically avoid in-panel titles
    
    # Color palette (colorblind-friendly, Wong et al. 2011)
    COLORS = {
        'black': (0, 0, 0),
        'orange': (230, 159, 0),
        'sky_blue': (86, 180, 233),
        'green': (0, 158, 115),
        'blue': (0, 114, 178),
        'deep_blue': (0, 78, 138),
        'teal': (0, 124, 136),
        'vermillion': (213, 94, 0),
        'purple': (204, 121, 167),
        'gray': (115, 115, 115),
    }
    
    @staticmethod
    def normalize_colors():
        """Convert RGB values from [0,255] to [0,1]"""
        return {k: tuple(v/255.0 for v in vals) 
                for k, vals in NatureStyleConfig.COLORS.items()}


class S2NFigure:
    """Generate Nature-style single-panel figures for S2N and G2S results"""
    
    def __init__(self, config=None):
        self.config = config or NatureStyleConfig()
        self.colors = NatureStyleConfig.normalize_colors()
        
        # Set matplotlib defaults for Nature
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for Nature publication"""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [self.config.FONT_NAME, 'Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['font.size'] = self.config.FONT_SIZE_AXIS
        
        # Important for PDF submission: embed fonts as TrueType (Type 42)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['savefig.transparent'] = False
        plt.rcParams['savefig.bbox'] = None
        
        # Remove spines and gridlines (Nature standard)
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.bottom'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = False
        plt.rcParams['axes.linewidth'] = 0.8
        
        # Tick styling
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.size'] = 3.0
        plt.rcParams['ytick.major.size'] = 3.0
        plt.rcParams['xtick.major.width'] = self.config.LINE_WIDTH
        plt.rcParams['ytick.major.width'] = self.config.LINE_WIDTH
        plt.rcParams['xtick.labelsize'] = self.config.FONT_SIZE_AXIS
        plt.rcParams['ytick.labelsize'] = self.config.FONT_SIZE_AXIS
    
    def load_plot_data(self, reg_csv, group_csv=None):
        """Derive both global and grouped errors from the same count pairs."""
        print(f"Loading regression data from: {reg_csv}")
        reg_df = pd.read_csv(reg_csv)
        group_df, _ = summarize_s2n_errors(reg_df)
        if group_csv is not None:
            print('Legacy group-total CSV is ignored; recomputing per-SU errors from regression points.')
        x = reg_df['true_count'].values
        y = reg_df['pred_count'].values
        print(f"Valid data points: {len(x)}")
        mae = np.mean(np.abs(x - y))
        r2 = self._compute_r2(x, y)
        print(f"R^2 = {r2:.4f}, MAE = {mae:.4f}\n")

        return x, y, group_df, r2, mae

    def load_g2s_regression_data(self, points_csv):
        """Load G2S test-point data and compute regression metrics"""

        print(f"Loading G2S regression data from: {points_csv}")
        points_df = pd.read_csv(points_csv)

        x = points_df['true_ppm'].values
        y = points_df['pred_ppm'].values

        if not len(x) or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('G2S shift pairs must be nonempty and finite; no rows are silently dropped.')
        if (x < 0).any() or (y < 0).any():
            raise ValueError('The nonnegative G2S density axes cannot display negative shifts.')

        print(f"Valid G2S data points: {len(x)}")

        mae = np.mean(np.abs(x - y))
        r2 = self._compute_r2(x, y)
        print(f"G2S R^2 = {r2:.4f}, MAE = {mae:.4f} ppm\n")

        return x, y, r2, mae

    def load_g2s_ablation_data(self, ablation_csv):
        """Load G2S hop-ablation metrics"""

        print(f"Loading G2S hop-ablation data from: {ablation_csv}")
        ablation_df = pd.read_csv(ablation_csv)
        return ablation_df.sort_values('k_hop').reset_index(drop=True)

    def _create_canvas(self, width_mm, height_mm, left, bottom, right, top):
        """Create a single-panel figure canvas"""

        mm_to_inch = 1.0 / 25.4
        fig_width_in = width_mm * mm_to_inch
        fig_height_in = height_mm * mm_to_inch

        fig = plt.figure(figsize=(fig_width_in, fig_height_in),
                         dpi=100, facecolor='white', edgecolor='none')
        ax = fig.add_axes([left, bottom, right - left, top - bottom])
        return fig, ax

    def create_regression_figure(self, x, y, r2, mae):
        """Create a standalone regression figure"""

        fig, ax = self._create_canvas(
            width_mm=self.config.REG_FIG_WIDTH_MM,
            height_mm=self.config.REG_FIG_HEIGHT_MM,
            left=0.17, bottom=0.14, right=0.84, top=0.95
        )
        self._plot_regression(ax, x, y, r2, mae, title='S2N test-set regression')
        return fig, ax

    def create_group_error_figure(self, group_df, use_mae_per_100c=False):
        """Create a standalone group-error bar figure"""

        fig, ax = self._create_canvas(
            width_mm=self.config.BAR_FIG_WIDTH_MM,
            height_mm=self.config.BAR_FIG_HEIGHT_MM,
            left=0.19, bottom=0.30, right=0.97, top=0.95
        )
        self._plot_group_errors(ax, group_df, use_mae_per_100c)
        return fig, ax

    def create_g2s_regression_figure(self, x, y, r2, mae):
        """Create a standalone G2S regression figure"""

        fig, ax = self._create_canvas(
            width_mm=self.config.REG_FIG_WIDTH_MM,
            height_mm=self.config.REG_FIG_HEIGHT_MM,
            left=0.17, bottom=0.14, right=0.84, top=0.95
        )
        self._plot_regression(
            ax, x, y, r2, mae,
            x_label='True chemical shift (ppm)',
            y_label='Predicted chemical shift (ppm)',
            mae_suffix=' ppm',
            title='G2S test-set regression'
        )
        return fig, ax

    def create_g2s_ablation_figure(self, ablation_df):
        """Create a standalone G2S hop-ablation figure"""

        fig, ax = self._create_canvas(
            width_mm=self.config.ABLATION_FIG_WIDTH_MM,
            height_mm=self.config.ABLATION_FIG_HEIGHT_MM,
            left=0.16, bottom=0.20, right=0.97, top=0.95
        )
        self._plot_hop_ablation(ax, ablation_df)
        return fig, ax
    
    def _plot_regression(self, ax, x, y, r2, mae,
                         x_label='True SU count',
                         y_label='Predicted SU count',
                         mae_suffix='',
                         title='Regression'):
        """Plot density-style regression on axes"""
        
        # Determine data range with small padding
        max_data_val = np.max([x, y])
        max_val = np.ceil(max_data_val / 10.0) * 10.0
        
        # Create 2D histogram (density plot)
        counts, xedges, yedges = np.histogram2d(
            x, y, 
            bins=70, 
            range=[[0, max_val], [0, max_val]]
        )
        
        # Avoid log(0) issues
        counts[counts == 0] = np.nan
        
        # Plot heatmap with log scale
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        finite_counts = counts[np.isfinite(counts)]
        vmax = np.max(finite_counts) if finite_counts.size > 0 else 1
        vmax = max(vmax, 1.0)

        im = ax.imshow(
            counts.T, 
            extent=extent, 
            origin='lower', 
            aspect='auto',
            cmap=self._build_nature_colormap(), 
            interpolation='nearest',
            norm=LogNorm(vmin=1, vmax=vmax, clip=True),
            rasterized=True
        )
        
        # Reference diagonal line (perfect prediction)
        ax.plot([0, max_val], [0, max_val], '--', 
               color=self.colors['black'], 
               linewidth=self.config.LINE_WIDTH, 
               zorder=2, 
               alpha=0.7)
        
        # Axis limits
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        
        # Labels
        ax.set_xlabel(x_label, 
                     fontsize=self.config.FONT_SIZE_LABEL,
                     fontname=self.config.FONT_NAME)
        ax.set_ylabel(y_label, 
                     fontsize=self.config.FONT_SIZE_LABEL,
                     fontname=self.config.FONT_NAME)
        if self.config.SHOW_TITLES:
            ax.set_title(title, 
                        fontsize=self.config.FONT_SIZE_LABEL,
                        fontname=self.config.FONT_NAME,
                        fontweight='normal', pad=8)
        
        # Tick formatting
        ax.tick_params(labelsize=self.config.FONT_SIZE_AXIS, which='both')
        ax.set_axisbelow(True)
        
        # Statistics annotation (FIXED: removed invalid 'none' boxstyle)
        textstr = f'R\u00b2 = {r2:.3f}\nMAE = {mae:.3f}{mae_suffix}'
        ax.text(0.05, 0.93, textstr, 
               transform=ax.transAxes,
               fontsize=self.config.FONT_SIZE_ANNOT,
               fontname=self.config.FONT_NAME,
               verticalalignment='top', 
               horizontalalignment='left')
        
        # Colorbar formatting
        cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.035, aspect=30)
        cbar.set_label('Sample count', 
                      fontsize=self.config.FONT_SIZE_LABEL,
                      fontname=self.config.FONT_NAME)
        cbar.ax.tick_params(labelsize=self.config.FONT_SIZE_AXIS)
        cbar.outline.set_linewidth(0.6)
    
    def _plot_group_errors(self, ax, group_df, use_mae_per_100c=False):
        """Plot group-wise mean absolute errors"""
        
        group_order = ['carbonyl', 'aromatic', 'unsaturated', 'aliphatic', 'hetero_terminal']
        group_labels = ['Carbonyl/nitrile', 'Aromatic', 'Unsaturated', 'Aliphatic', 'Heteroatom']
        
        metric = 'per_su_mae_per_100C' if use_mae_per_100c else 'per_su_mae'
        ylabel = 'Per-SU MAE (per 100 C)' if use_mae_per_100c else 'Per-SU count MAE'
        if metric not in group_df:
            raise ValueError('Group bars require per-SU errors computed from regression points, not group totals.')
        
        # Extract values
        vals = []
        for grp in group_order:
            row = group_df[group_df['group'] == grp]
            if len(row) > 0:
                vals.append(row[metric].values[0])
            else:
                vals.append(np.nan)
        vals = np.array(vals)
        
        # Bar colors (colorblind-friendly)
        bar_colors_rgb = [
            self.colors['orange'],
            self.colors['blue'],
            self.colors['green'],
            self.colors['vermillion'],
            self.colors['purple']
        ]
        
        # Create bars
        x_pos = np.arange(len(group_labels))
        bars = ax.bar(x_pos, vals, width=self.config.BAR_WIDTH, 
                     edgecolor=self.colors['black'],
                     linewidth=0.55, 
                     color=bar_colors_rgb,
                     zorder=2)
        
        # Value labels on top of bars
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if np.isfinite(val):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, 
                       height + np.nanmax(vals) * 0.025,
                       f'{val:.2f}', 
                       ha='center', va='bottom',
                       fontsize=self.config.FONT_SIZE_TICK,
                       fontname=self.config.FONT_NAME,
                       color=self.colors['black'],
                       fontweight='normal')
        
        # X-axis formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_labels, 
                          rotation=22, 
                          ha='right',
                          fontsize=self.config.FONT_SIZE_AXIS_CAT,
                          fontname=self.config.FONT_NAME)
        ax.margins(x=0.05)
        
        # Labels
        ax.set_ylabel(ylabel, 
                     fontsize=self.config.FONT_SIZE_LABEL,
                     fontname=self.config.FONT_NAME)
        if self.config.SHOW_TITLES:
            ax.set_title('Group-wise prediction error', 
                        fontsize=self.config.FONT_SIZE_LABEL,
                        fontname=self.config.FONT_NAME,
                        fontweight='normal', pad=8)
        
        # Y-axis formatting
        y_max = np.nanmax(vals)
        ax.set_ylim(0, y_max * 1.14)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='x', labelsize=self.config.FONT_SIZE_AXIS_CAT, which='both', pad=2)
        ax.tick_params(axis='y', labelsize=self.config.FONT_SIZE_AXIS, which='both')
        ax.set_axisbelow(True)
        
        # Add subtle horizontal gridlines (optional, for readability)
        ax.yaxis.grid(False, which='both')

    def _plot_hop_ablation(self, ax, ablation_df):
        """Plot G2S hop-ablation MAE and RMSE"""

        x = ablation_df['k_hop'].values
        mae = ablation_df['mae'].values
        rmse = ablation_df['rmse'].values

        ax.plot(
            x, mae,
            color=self.colors['blue'],
            marker='o',
            markersize=5.0,
            linewidth=1.4,
            label='MAE'
        )
        ax.plot(
            x, rmse,
            color=self.colors['vermillion'],
            marker='s',
            markersize=4.8,
            linewidth=1.4,
            linestyle='--',
            label='RMSE'
        )

        ax.set_xlabel('Subgraph radius k-hop',
                     fontsize=self.config.FONT_SIZE_LABEL,
                     fontname=self.config.FONT_NAME)
        ax.set_ylabel('Error (ppm)',
                     fontsize=self.config.FONT_SIZE_LABEL,
                     fontname=self.config.FONT_NAME)

        ax.set_xlim(np.min(x) - 0.15, np.max(x) + 0.15)
        ax.set_xticks(x)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        y_all = np.concatenate([mae, rmse])
        y_min = np.min(y_all)
        y_max = np.max(y_all)
        y_span = max(y_max - y_min, 0.2)
        ax.set_ylim(max(0, y_min - 0.10 * y_span), y_max + 0.08 * y_span)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.tick_params(axis='x', labelsize=self.config.FONT_SIZE_AXIS, which='both')
        ax.tick_params(axis='y', labelsize=self.config.FONT_SIZE_AXIS, which='both')
        ax.set_axisbelow(True)
        ax.legend(
            loc='upper right',
            frameon=False,
            fontsize=self.config.FONT_SIZE_LABEL,
            handlelength=2.0
        )
    
    @staticmethod
    def _compute_r2(x, y):
        """Calculate R^2 coefficient of determination"""
        ss_res = np.sum((x - y) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    
    @staticmethod
    def _build_nature_colormap(n=256):
        """Create a darker, higher-contrast sequential colormap for publication density maps."""
        anchor = [
            (0.00, (0.85, 0.92, 0.98)),
            (0.08, (0.74, 0.87, 0.97)),
            (0.22, (0.56, 0.79, 0.94)),
            (0.40, (0.27, 0.60, 0.87)),
            (0.62, (0.07, 0.39, 0.71)),
            (0.82, (0.02, 0.22, 0.47)),
            (1.00, (0.00, 0.10, 0.24)),
        ]
        return LinearSegmentedColormap.from_list('nature_density', anchor, N=n)
    
    def save_figure(self, fig, output_dir, base_name, dpi=None):
        """Export one figure in multiple formats (Nature-compliant)"""
        
        if dpi is None:
            dpi = self.config.DPI
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Exporting figure (DPI: {dpi})...")
        print(f"{'='*60}")
        
        # Vector PDF (primary format for Nature)
        pdf_path = os.path.join(output_dir, f'{base_name}.pdf')
        fig.savefig(pdf_path, format='pdf', dpi=dpi,
                   facecolor='white', edgecolor='none')
        print(f"[OK] PDF saved: {pdf_path}")
        
        # High-resolution PNG (for presentations)
        png_path = os.path.join(output_dir, f'{base_name}.png')
        fig.savefig(png_path, format='png', dpi=dpi,
                   facecolor='white', edgecolor='none')
        print(f"[OK] PNG saved: {png_path}")

        # TIFF for journal workflows
        tif_path = os.path.join(output_dir, f'{base_name}.tif')
        fig.savefig(tif_path, format='tiff', dpi=dpi,
                   pil_kwargs={'compression': 'tiff_lzw'},
                   facecolor='white', edgecolor='none')
        print(f"[OK] TIF saved: {tif_path}")
        svg_path = os.path.join(output_dir, f'{base_name}.svg')
        fig.savefig(svg_path, format='svg', dpi=dpi,
                    facecolor='white', edgecolor='none')
        print(f"[OK] SVG saved: {svg_path}")
        
        print(f"\n{'='*60}")
        print(f"Export Summary:")
        print(f"  Output directory: {output_dir}")
        print(f"  Export DPI: {dpi}")
        print(f"  Font: {self.config.FONT_NAME} (Nature standard)")
        print(f"  Font embedding: TrueType (Type 42)")
        print(f"  Titles shown: {'Yes' if self.config.SHOW_TITLES else 'No'}")
        print(f"{'='*60}\n")


def infer_k_tag(path):
    """Infer k-hop tag like k3 from a file path, if present."""

    match = re.search(r'(k\d+)', os.path.normpath(path))
    return match.group(1) if match else None


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Generate Nature-style single-panel figures for S2N and G2S results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default S2N figures
  python plot.py --figure_set s2n --repo_root .

  # G2S test regression (default k3 path)
  python plot.py --figure_set g2s_regression --repo_root . --out_dir ./Nature_figures

  # G2S hop ablation
  python plot.py --figure_set g2s_ablation --repo_root . --out_dir ./Nature_figures

  # Generate everything together
  python plot.py --figure_set all --repo_root . --out_dir ./Nature_figures
        """)
    
    parser.add_argument('--figure_set', type=str,
                       choices=['s2n', 'g2s_regression', 'g2s_ablation', 'all'],
                       default='s2n',
                       help='Which figure set to generate')
    parser.add_argument('--repo_root', type=str, default=SCRIPT_DIR,
                       help='Root directory for data search')
    parser.add_argument('--reg_csv', type=str,
                       help='Path to S2N regression CSV')
    parser.add_argument('--group_csv', type=str,
                       help='Deprecated group-total CSV argument; ignored in favor of per-SU errors')
    parser.add_argument('--g2s_points_csv', type=str,
                       help='Path to G2S test_points CSV')
    parser.add_argument('--g2s_ablation_csv', type=str,
                       help='Path to G2S hop-ablation metrics CSV')
    parser.add_argument('--out_dir', type=str, default='./output_figures',
                       help='Output directory for figures')
    parser.add_argument('--use_mae_per_100c', action='store_true',
                       help='Use MAE per 100 C instead of raw MAE')
    parser.add_argument('--dpi', type=int, default=600,
                       help='Raster export resolution (default: 600 dpi)')
    parser.add_argument('--reg_width_mm', type=float, default=75,
                       help='Regression figure width in mm (default: 75, about 7.5 cm)')
    parser.add_argument('--reg_height_mm', type=float, default=72,
                       help='Regression figure height in mm')
    parser.add_argument('--bar_width_mm', type=float, default=75,
                       help='Bar-chart figure width in mm (default: 75, about 7.5 cm)')
    parser.add_argument('--bar_height_mm', type=float, default=58,
                       help='Bar-chart figure height in mm')
    parser.add_argument('--ablation_width_mm', type=float, default=75,
                       help='Ablation-plot figure width in mm')
    parser.add_argument('--ablation_height_mm', type=float, default=58,
                       help='Ablation-plot figure height in mm')
    parser.add_argument('--show_titles', action='store_true',
                       help='Show in-panel titles (off by default for submission figures)')
    parser.add_argument('--show', action='store_true',
                       help='Display figures interactively after export')
    
    args = parser.parse_args()
    args.repo_root = os.path.normpath(args.repo_root)

    if args.reg_csv is None:
        args.reg_csv = os.path.join(
            args.repo_root, 'checkpoints_s2n_v1', 'eval', 'plot_data', 'test_regression_points.csv'
        )
    if args.g2s_points_csv is None:
        args.g2s_points_csv = os.path.join(
            args.repo_root, 'checkpoints_g2s_ablation', 'k3', 'eval', 'test_points.csv'
        )
    if args.g2s_ablation_csv is None:
        args.g2s_ablation_csv = os.path.join(
            args.repo_root, 'checkpoints_g2s_ablation', 'hop_ablation_key_metrics.csv'
        )

    print("Checking input files...")
    required_files = []
    if args.figure_set in ('s2n', 'all'):
        required_files.extend([
            (args.reg_csv, 'S2N regression CSV'),
        ])
    if args.figure_set in ('g2s_regression', 'all'):
        required_files.append((args.g2s_points_csv, 'G2S test_points CSV'))
    if args.figure_set in ('g2s_ablation', 'all'):
        required_files.append((args.g2s_ablation_csv, 'G2S hop-ablation CSV'))

    for fpath, name in required_files:
        if not os.path.exists(fpath):
            print(f"[ERROR] {name} not found: {fpath}")
            return
        print(f"[OK] {name} found: {fpath}")

    print("\nCreating figures...")
    config = NatureStyleConfig()
    config.REG_FIG_WIDTH_MM = args.reg_width_mm
    config.REG_FIG_HEIGHT_MM = args.reg_height_mm
    config.BAR_FIG_WIDTH_MM = args.bar_width_mm
    config.BAR_FIG_HEIGHT_MM = args.bar_height_mm
    config.ABLATION_FIG_WIDTH_MM = args.ablation_width_mm
    config.ABLATION_FIG_HEIGHT_MM = args.ablation_height_mm
    config.DPI = max(args.dpi, 300)
    config.SHOW_TITLES = args.show_titles

    figmaker = S2NFigure(config)
    generated_figs = []

    if args.figure_set in ('s2n', 'all'):
        x, y, group_df, r2, mae = figmaker.load_plot_data(args.reg_csv, args.group_csv)
        reg_fig, _ = figmaker.create_regression_figure(x, y, r2, mae)
        bar_fig, _ = figmaker.create_group_error_figure(
            group_df,
            use_mae_per_100c=args.use_mae_per_100c
        )
        figmaker.save_figure(reg_fig, args.out_dir, base_name='S2N_regression', dpi=config.DPI)
        figmaker.save_figure(bar_fig, args.out_dir, base_name='S2N_group_error', dpi=config.DPI)
        generated_figs.extend([reg_fig, bar_fig])

    if args.figure_set in ('g2s_regression', 'all'):
        gx, gy, gr2, gmae = figmaker.load_g2s_regression_data(args.g2s_points_csv)
        g2s_reg_fig, _ = figmaker.create_g2s_regression_figure(gx, gy, gr2, gmae)
        g2s_tag = infer_k_tag(args.g2s_points_csv)
        g2s_reg_name = f'G2S_test_regression_{g2s_tag}' if g2s_tag else 'G2S_test_regression'
        figmaker.save_figure(g2s_reg_fig, args.out_dir, base_name=g2s_reg_name, dpi=config.DPI)
        generated_figs.append(g2s_reg_fig)

    if args.figure_set in ('g2s_ablation', 'all'):
        ablation_df = figmaker.load_g2s_ablation_data(args.g2s_ablation_csv)
        ablation_fig, _ = figmaker.create_g2s_ablation_figure(ablation_df)
        figmaker.save_figure(ablation_fig, args.out_dir, base_name='G2S_hop_ablation', dpi=config.DPI)
        generated_figs.append(ablation_fig)

    print("\n[OK] Figure generation completed successfully.")
    if args.show:
        plt.show()
    else:
        for fig in generated_figs:
            plt.close(fig)


if __name__ == '__main__':
    main()
