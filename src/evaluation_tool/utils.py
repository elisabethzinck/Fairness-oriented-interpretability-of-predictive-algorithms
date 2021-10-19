import numpy as np
import seaborn as sns
import matplotlib as plt
import colorsys
from matplotlib import transforms
from statsmodels.stats.proportion import proportion_confint

##############################################
#                Functions 
##############################################
def flatten_list(t):
    """Flattens a list of list"""
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

def cm_vals_to_matrix(TP, FN, FP, TN):
    """Converts values of TP, FN, FP and TN into a confusion matrix
    
    Args:
        TP (int): True Positives
        FN (int): False Negatives
        FP (int): False Positives 
        TN (int): True Negatives

    Returns: 
        Numpy array of shape (2,2) 
    """
    cm_matrix = np.array(
        [
            [TP, FN], 
            [FP, TN]
        ]
    )
    return cm_matrix

def cm_dict_to_matrix(cm_dict):
    """Convert a confusion dict to confusion matrix
    
    Args:
        cm_dict (dict): Confusion dict with keys TP, FP, TN, FN

    Returns: 
        Numpy array of shape (2,2) 
    """
    cm_matrix = np.array(
        [
            [cm_dict['TP'], cm_dict['FN']], 
            [cm_dict['FP'], cm_dict['TN']]
        ]
    )
    return cm_matrix

def cm_matrix_to_dict(cm_matrix):
    """
    
    Convert a 2x2 confusion matrix to named dict

    Confusion matrix must be of form [[TP, FN], [FN, TN]]
    """
    TN, FP, FN, TP = cm_matrix.ravel()
    cm_dict = {'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN}
    return cm_dict

def abs_percentage_tick(x, pos):
    """Return absolute percentage value w/ % as string"""
    return str(round(abs(x))) + '%'

def round_func(x, base = 5):
    """Costum round function, which rounds to the nearest base. Default base = 5"""
    return(round(x/base)*base)

def flip_dataframe(df, new_colname = 'index'):
    """Flips table such that first row becomes columns"""
    colnames = [new_colname] + df.iloc[:,0].tolist()
    df = df.T.reset_index()
    df.columns = colnames
    df = df.iloc[1:, :]
    return df

def custom_palette(n_colors = 1, specific_col_idx = None):
    """Returns a custom palette of n_colors (max 10 colors)
    
        The color palette have been created using this link: 
        https://coolors.co/f94d50-f3722c-f8961e-f9844a-f9c74f-a1c680-3a9278-7ab8b6-577590-206683
        
        Args:
            n_color: The number of desired colors max 10. Defaults to 1.
            specific_col_idx: list of desired color indexes. Defaults to None.
            
        Returns: 
            A custom seaborn palette of n_colors length 
    """
    colors =  ["f94d50","f3722c","f8961e","f9844a","f9c74f","a1c680","3a9278","7ab8b6","577590","206683"]




    max_colors = len(colors)
    assert n_colors < max_colors, "n_colors must be less than 10"    
    
    if specific_col_idx is None:   
        idx = np.ceil(np.linspace(start=0, stop=max_colors-1, num=n_colors))
        col_idx = [int(idx[i]) for i in range(n_colors)]
    else:
        col_idx = specific_col_idx
        n_colors = len(col_idx)
    
    hex_colors = [f'#{colors[col_idx[i]]}' for i in range(n_colors)]

    return sns.color_palette(hex_colors)

def add_colors_with_stripes(ax, color_dict, color_variable):
    """Add colors to barplot including striped colors.
    
    Args:
        ax (AxesSubplot): ax of plot to color
        color_dict (dict): Dict of colors with keys as color group
        color_variable (pd.Series): Series of color groups used in bars. 
            Each item in series is a list. 
    """
    # Adjust colors to match other seaborn plots
    def desaturate(color, prop = 0.75):
        """Desaturate color just like in default seaborn plot"""
        h,l,s = colorsys.rgb_to_hls(*color)
        s *= prop
        new_color = colorsys.hls_to_rgb(h, l, s)
        return new_color
    muted_colors = {k:desaturate(col) for (k,col) in color_dict.items()}
    bar_colors = [[muted_colors[grp] for grp in grp_list] for grp_list in color_variable]

    # Set colors of bars.
    plt.rcParams['hatch.linewidth'] = 8 # Controls thickness of stripes
    for bar, var in zip(ax.containers[0], color_variable):
        if len(var) == 1:
            col = muted_colors[var[0]]
            bar.set_facecolor(col)
        elif len(var) == 2:
            col0 = muted_colors[var[0]]
            col1 = muted_colors[var[1]]
            bar.set_facecolor(col0)
            bar.set_edgecolor(col1)
            bar.set_hatch('/')
            bar.set_linewidth(0) # No outline around bar when striped
        else:
            raise IndexError('Cannot use > 2 colors for stripes in barplot')

def get_alpha_weights(w_fp):
    """Return alpha weight for each rate"""
    c = 0.2 # Factor added to make colors stronger
    if w_fp == 0.5:
        alpha_weights = {'FPR': 1, 'FNR': 1, 'FDR': 1, 'FOR': 1, 'WMR': 1}
    elif w_fp > 0.5:
        alpha_weights = {'FPR': 1, 'FNR': 1+c-w_fp, 'FDR':1, 'FOR':1+c-w_fp, 'WMR': 1}
    else: 
        alpha_weights = {'FPR': c+w_fp, 'FNR': 1, 'FDR':c+w_fp, 'FOR':1, 'WMR': 1}
    return alpha_weights

def error_bar(ax, plot_df, bar_mid, orientation = "v"):
    """Draws error bars on ax with barplot.
    
    Args: 
        ax(matplotlib.axes): ax with barplot 
        plot_df(pandas data frame): must include columns "conf_lwr" and "conf_upr"
        bar_mid(float): mid point of bar to put error bar on 
        orientation: 'v' or 'h' for vertical or horizontal 
    """
    # input check
    assert orientation in ["v", "h"], "Choose orientation 'v' or 'h'"
    assert "conf_lwr" in plot_df.columns, 'column "conf_lwr" must be in data' 
    assert "conf_upr" in plot_df.columns, 'column "conf_upr" must be in data'

    if orientation == 'v':
        ax.vlines(x=bar_mid, 
                ymin=plot_df.conf_lwr,
                ymax=plot_df.conf_upr,
                colors = (58/255, 58/255, 58/255),
                linewidth = 2)
        ax.hlines(y=[plot_df.conf_lwr, plot_df.conf_upr], 
                xmin=bar_mid-0.15,
                xmax=bar_mid+0.15,
                colors = (58/255, 58/255, 58/255),
                linewidth = 2)
                
    elif orientation == 'h':
        ax.hlines(y=bar_mid, 
                xmin=plot_df.conf_lwr,
                xmax=plot_df.conf_upr,
                colors = (58/255, 58/255, 58/255),
                linewidth = 2)
        ax.vlines(x=[plot_df.conf_lwr, plot_df.conf_upr], 
                ymin=bar_mid-0.15,
                ymax=bar_mid+0.15,
                colors = (58/255, 58/255, 58/255),
                linewidth = 2)

def desaturate(color, prop = 0.75):
        """Desaturate color just like in default seaborn plot"""
        h,l,s = colorsys.rgb_to_hls(*color)
        s *= prop
        new_color = colorsys.hls_to_rgb(h, l, s)
        return new_color

def value_counts_df(df, col_name):
    """pd.value_counts() for dataframes
    
    Args:
        df (pd.DataFrame): Dataframe containing column to use value_counts on
        col_names (str): Name of column to count values for. Must be present in df. 

    Returns:
        Dataframe with column names `col_name` and `n`.
    
    """
    if col_name not in df.columns:
        raise ValueError(f'Supplied col_name {col_name} is not present in dataframe.')

    count_df = df[col_name].value_counts().to_frame().reset_index()
    count_df.columns = [col_name, 'n']

    return count_df

def label_case(snake_case):
    "Replace underscore with spaces and capitalize first letter of string"
    return snake_case.replace("_", ' ').capitalize()

def format_text_layer_1(ax, x, y, text_list, color_list, font_sizes, font_weights):
    """Plots a list of words with specific colors, sizes and font 
    weights on a provided axis 
    Function inspired by: https://matplotlib.org/2.0.2/examples/text_labels_and_annotations/rainbow_text.html
    Args:
        ax(matplotlib axis): axis to plot text on 
        x(float): x-coordinate of text start
        y(float): y-coordinate of text start
        text_list(list of strings): List with words to plot 
        color_list(list): List of colors for each word in text_list
        font_sizes(list): List of font sizes for each word in text_list
        font_sizes(list): List of font weigths for each word in text_list
    """
    t = ax.transData
    canvas = ax.figure.canvas
    
    for s, c, f, w in zip(text_list, color_list, font_sizes, font_weights):
        text = ax.text(x, y, s + " ", color=c, 
                       fontsize = f, weight = w, transform=t)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

def wilson_confint(n_successes, n, side):
    """Calculate Wilson proportion CI
    
    Args: 
        n_successes (int): Count
        n (int): Number of observations
        side ({'lwr', 'upr'}): Return lower or upper side of interval
    """
    conf = proportion_confint(
        count = n_successes,
        nobs = n,
        method = "wilson")
    
    if side == 'lwr':
        return conf[0]
    elif side == 'upr':
        return conf[1]
    else:
        raise ValueError(f"`side` must be in ['lwr', 'upr']. You supplied `side` = {side}")
        return None

def extract_cm_values(df, grp):
    """Extracts TP, TN, FP and TN from long format confusion matrix
    
    Args: 
        df(Dataframe): long format confusion matrix as returned 
                        by get_confusion_matrix() in layered_tool
        grp(str): sensitive group name 
    """
    cm_data = df.query(f"a=='{grp}' & type_obs in ['TP', 'FN', 'FP', 'TN']")
    TP, FN, FP, TN = cm_data.number_obs.to_list()

    return TP, FN, FP, TN

def static_split(string, pattern, n_elem):
    "Same as str.split() except `n_elem`elements are always returned"
    tmp = string.split(pattern, maxsplit = n_elem-1) # n_elem = n_split + 1
    n_remaining = n_elem - len(tmp)
    out = tmp + [None]*n_remaining
    return out
    
################################################
#             lambda functions
################################################
N_pos = lambda x: np.count_nonzero(x)
N_neg = lambda x: len(x)-np.count_nonzero(x)
frac_pos = lambda x: (np.count_nonzero(x)/len(x))
frac_neg = lambda x: (len(x)-np.count_nonzero(x))/len(x)



if __name__ == '__main__':
    n_colors = 3
    cp = custom_palette(n_colors = n_colors)