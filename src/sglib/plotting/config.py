"""
Centralized plotting configuration for SGLib.

All plotting functions use these defaults, which can be overridden
by environment variables or user settings.
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

# ============================================================================
# COLOR SCHEMES
# ============================================================================

COLORS = {
    # Primary data colors
    'observed': '#2C3E50',        # Dark slate gray
    'ensemble_median': '#E67E22',  # Orange
    'ensemble_mean': '#E74C3C',    # Red-orange

    # Ensemble uncertainty
    'ensemble_fill': '#F39C12',    # Light orange (for fill_between)
    'ensemble_members': '#95A5A6', # Gray (for individual realizations)

    # Quantile colors (from darkest to lightest)
    'quantile_dark': '#D35400',    # Dark orange
    'quantile_mid': '#E67E22',     # Medium orange
    'quantile_light': '#F39C12',   # Light orange

    # Drought-specific
    'drought_moderate': '#F39C12', # Orange (SSI < -1)
    'drought_severe': '#E74C3C',   # Red (SSI < -2)
    'drought_extreme': '#C0392B',  # Dark red (SSI < -3)

    # Grid and accents
    'grid': '#BDC3C7',             # Light gray
    'zero_line': '#34495E',        # Dark gray
}

# ============================================================================
# STYLE SETTINGS
# ============================================================================

STYLE = {
    # Line styles
    'observed_linewidth': 2.0,
    'ensemble_linewidth': 1.5,
    'member_linewidth': 0.8,
    'member_alpha': 0.3,

    # Fill styles
    'fill_alpha': 0.3,

    # Marker styles
    'observed_marker': 'o',
    'observed_markersize': 4,

    # Grid
    'grid_alpha': 0.3,
    'grid_linestyle': '--',
    'grid_linewidth': 0.5,
}

# ============================================================================
# LAYOUT SETTINGS
# ============================================================================

LAYOUT = {
    # Default figure sizes (width, height) in inches
    'default_figsize': (10, 6),
    'wide_figsize': (12, 5),
    'square_figsize': (8, 8),
    'validation_figsize': (10, 12),

    # DPI settings
    'default_dpi': 100,      # Display
    'save_dpi': 300,         # Saved figures

    # Font sizes
    'title_fontsize': 14,
    'label_fontsize': 12,
    'tick_fontsize': 10,
    'legend_fontsize': 10,

    # Spacing
    'tight_layout': True,
}

# ============================================================================
# LABEL DEFAULTS
# ============================================================================

LABELS = {
    'flow_units': {
        'cms': 'Streamflow (m³/s)',
        'cfs': 'Streamflow (ft³/s)',
        'mgd': 'Streamflow (MGD)',
        'maf': 'Streamflow (MAF)',
    },

    'timestep_labels': {
        'daily': 'Day of Year',
        'weekly': 'Week of Year',
        'monthly': 'Month',
        'annual': 'Year',
    },

    'month_labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
}

# ============================================================================
# MATPLOTLIB RCPARAMS
# ============================================================================

def apply_plotting_style():
    """Apply global matplotlib style settings for SGLib plots."""
    plt.rcParams.update({
        'figure.figsize': LAYOUT['default_figsize'],
        'figure.dpi': LAYOUT['default_dpi'],
        'savefig.dpi': LAYOUT['save_dpi'],
        'savefig.bbox': 'tight',

        'font.size': LAYOUT['tick_fontsize'],
        'axes.titlesize': LAYOUT['title_fontsize'],
        'axes.labelsize': LAYOUT['label_fontsize'],
        'xtick.labelsize': LAYOUT['tick_fontsize'],
        'ytick.labelsize': LAYOUT['tick_fontsize'],
        'legend.fontsize': LAYOUT['legend_fontsize'],

        'axes.grid': True,
        'grid.alpha': STYLE['grid_alpha'],
        'grid.linestyle': STYLE['grid_linestyle'],
        'grid.linewidth': STYLE['grid_linewidth'],

        'axes.spines.top': False,
        'axes.spines.right': False,

        'legend.frameon': False,
        'legend.loc': 'best',
    })

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_color_palette() -> Dict[str, str]:
    """Return a copy of the color palette."""
    return COLORS.copy()

def get_style_config() -> Dict[str, Any]:
    """Return a copy of the style configuration."""
    return STYLE.copy()

def get_layout_config() -> Dict[str, Any]:
    """Return a copy of the layout configuration."""
    return LAYOUT.copy()

def get_labels() -> Dict[str, Any]:
    """Return a copy of the label configurations."""
    return LABELS.copy()