from math import log, sqrt, exp
import matplotlib.pyplot as plt
from scipy.stats import norm

def r_to_z(r):
    return log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    z = r_to_z(r)
    se = 1.0 / sqrt(n - 3)
    z_crit = norm.ppf((1 + alpha)/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    return (z_to_r(lo), z_to_r(hi))

# https://cmdlinetips.com/2019/10/how-to-make-a-plot-with-two-different-y-axis-in-python-with-matplotlib/
def twin_plot(x_axis_values : pd.Series, 
              series_1_values : pd.Series, 
              series_2_values : pd.Series, 
              x_label : str, 
              series_1_label: str, 
              series_2_label : str, 
              series_1_color : str = "red", 
              series_2_color : str = "blue", 
              series_2_x_shift : int = 0, 
              figsize : tuple = (12, 8), 
              title : str = ""):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_axis_values, series_1_values, color=series_1_color, marker="o")
    ax.set_xlabel(x_label, fontsize = 14)
    ax.set_ylabel(series_1_label, color=series_1_color, fontsize=14)


    ax2=ax.twinx()
    ax2.plot(x_axis_values + series_2_x_shift, series_2_values,color=series_2_color,marker="o")
    ax2.set_ylabel(series_2_label,color=series_2_color,fontsize=14)
    
    plt.title(title)
    
    plt.show()

def climate_title(series_1 : pd.Series, series_2 : pd.Series, alpha : float = 0.95, prefix : str = "") -> str:
    r = series_1.corr(series_2)
    n = len(series_1)
    
    lo_r, hi_r = r_confidence_interval(r, alpha, n)
    
    return f"{prefix}The correlation is {r:.1%} with {alpha:.0%} confidence that the correlation lies between {lo_r:.1%} and {hi_r:.1%}"
    
def climate_plot(series_1_feature : str, 
                 series_2_feature : str, 
                 x_axis_values : pd.Series = df_climate_change.index, 
                 title : str = "", 
                 x_label : str = "Year", 
                 series_2_x_shift : int = 0, 
                 figsize : tuple = (12, 8)):
    
    series_1_values = df_climate_change[series_1_feature]
    series_2_values = df_climate_change[series_2_feature]
    
    series_1_label = df_features.at[series_1_feature, "Feature Name"] + " (" + df_features.at[series_1_feature, "Unit"] + ")"
    series_2_label = df_features.at[series_2_feature, "Feature Name"] + " (" + df_features.at[series_2_feature, "Unit"] + ")"
    
    twin_plot(x_axis_values, 
              series_1_values, 
              series_2_values, 
              x_label, 
              series_1_label, 
              series_2_label, 
              series_2_x_shift = series_2_x_shift, 
              figsize=figsize, 
              title=climate_title(series_1 = series_1_values, series_2 = series_2_values, prefix=title))       