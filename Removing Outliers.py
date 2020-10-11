# %% imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
#from scipy import stats
print("imports complete")

# %% [markdown]
# # Outliers
# 
# A boxplot showing the median and inter-quartile ranges is a good way to visualise a distribution, especially when 
# the data contains outliers. The meaning of the various aspects of a box plot can be visualised as follows -
# 
# ![](images/IQR.png)

# %% [markdown]
# ## Generating some data
# We are going to need some test data to explore the issues around outliers


# %% [markdown]
# ### Function Definition
#
# The generate() function below (taken from Stack Overflow) will generate a list of floats
# with a given median that contains outliers (values a long way from the median) which we can
# use to explore the concept.

# %% Generate function
def generate(median=630, err=12, outlier_err=100, size=80, outlier_size=10):
    np.random.seed(median)
    errs = err * np.random.rand(size) * np.random.choice((-1, 1), size)
    data = median + errs

    lower_errs = outlier_err * np.random.rand(outlier_size)
    lower_outliers = median - err - lower_errs

    upper_errs = outlier_err * np.random.rand(outlier_size)
    upper_outliers = median + err + upper_errs

    data = np.concatenate((data, lower_outliers, upper_outliers))
    np.random.shuffle(data)

    return data

# %% [markdown]
# ### Function Testing
#
# Let's get the results of generate() into a DataFrame so we can take a look at the output ...

# %% Test the generate function out ...
df_test = pd.DataFrame(generate())
df_test.head()

# %% [markdown]
# ## Creating some meaningful data
#
# The following creates a dataframe with 3 columns with means of 630, 740 and 220 which contain ouutlying values

# %% Create the dataframe
df = pd.DataFrame({'Col0': generate(median=630), 'Col1': generate(median=740), 'Col2': generate(median=220)})
df.head()

# %% [markdown]
# ## Working with outliers
#
# Let's start by taking another look at the explanation - 
# 
# ![](images\IQR.png)

# %% [markdown]
# ### Helpder functions
#
# The functions below looks at a column of values within a data frame and calculates
# the 1st and 3rd quartiles, the inter-quartile range and the minimum and maximum
# Any value outside of the miniumum and maximum is an outlier

# %% Define fuctions to calculate the iqr values and then apply them to remove outliers...
# (modified from http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)
def get_iqr_values(df_in, col_name):
    median = df_in[col_name].median()
    q1 = df_in[col_name].quantile(0.25) # 25th percentile / 1st quartile
    q3 = df_in[col_name].quantile(0.75) # 7th percentile / 3rd quartile
    iqr = q3-q1 #Interquartile range
    minimum  = q1-1.5*iqr # The minimum value or the |- marker in the box plot
    maximum = q3+1.5*iqr # The maximum value or the -| marker in the box plot
    return median, q1, q3, iqr, minimum, maximum

def get_iqr_text(df_in, col_name):
    median, q1, q3, iqr, minimum, maximum = get_iqr_values(df_in, col_name)
    text = f"median={median:.2f}, q1={q1:.2f}, q3={q3:.2f}, iqr={iqr:.2f}, minimum={minimum:.2f}, maximum={maximum:.2f}"
    return text

def remove_outliers(df_in, col_name):
    _, _, _, _, minimum, maximum = get_iqr_values(df_in, col_name)
    df_out = df_in.loc[(df_in[col_name] > minimum) & (df_in[col_name] < maximum)]
    return df_out

def count_outliers(df_in, col_name):
    _, _, _, _, minimum, maximum = get_iqr_values(df_in, col_name)
    df_outliers = df_in.loc[(df_in[col_name] <= minimum) | (df_in[col_name] >= maximum)]
    return df_outliers.shape[0]

def box_and_whisker(df_in, col_name):
    title = get_iqr_text(df_in, col_name)
    sns.boxplot(df_in[col_name])
    plt.title(title)
    plt.show()

print("functions defined")

# %% [markdown]
# ## Visualising the data
#
# sns.boxplot is used to visualise our 3 columns of data; the outliers are the dots that lie 
# outside of the |- and -| whiskers ...

# %% Plot the graphs
box_and_whisker(df, 'Col0')
box_and_whisker(df, 'Col1')
box_and_whisker(df, 'Col2')

_, _, _, _, minimum_Col1_before, maximum_Col1_before = get_iqr_values(df, 'Col1')

# %% Count the outliers in the original data frame
print(f"Col0 has {count_outliers(df, 'Col0')} outliers")
print(f"Col1 has {count_outliers(df, 'Col1')} outliers")
print(f"Col2 has {count_outliers(df, 'Col2')} outliers")

# %% [markdown]
# ## Removing the outliers
#
# Having done all the heavy lifting in the helpder functions we can now go ahead and remove the rows from the outliers
# outside of the |- and -| whiskers ...

# %% remove the outliers
print(f"rows before removing: {df.shape[0]}")
df = remove_outliers(df, 'Col0')
df = remove_outliers(df, 'Col1')
df = remove_outliers(df, 'Col2')
print(f"rows after removing: {df.shape[0]}")

# %% [markdown]
# ## Visualise the result
#
# Let's have a look at the end-result. Here is something very strange though, our data still appears to have outliers!
box_and_whisker(df, 'Col0')
box_and_whisker(df, 'Col1')
box_and_whisker(df, 'Col2')

_, _, _, _, minimum_Col1_after, maximum_Col1_after = get_iqr_values(df, 'Col1')

# %% [markdown]
# ## Explain the result
#
# The reason that Col0 and Col1 still appear to have outliers is that we removed the outliers based on the 
# minimum and maximum of the dataframe before we modified it with 
# 
#    df = remove_outliers(df, 'Col0')
#    df = remove_outliers(df, 'Col1')
#    df = remove_outliers(df, 'Col2')
# 
# Once the data has been changed some values will be retained that were close to the original boundaries but 
# after the modification the boundaries will change leaving some of the values outside of the new boundaries

# %% Explain the results in data ...
print(f"Col1 original boundaries: minium={minimum_Col1_before:.2f}, maximum={maximum_Col1_before:.2f}")
print(f"Col1 new minimum and maximum values: minium={df['Col1'].min():.2f}, maximum={df['Col1'].max():.2f}")
print(f"Col1 new boundaries: minium={minimum_Col1_after:.2f}, maximum={maximum_Col1_after:.2f}")
print("")
print(f"Col0 has {count_outliers(df, 'Col0')} outliers")
print(f"Col1 has {count_outliers(df, 'Col1')} outliers")
print(f"Col2 has {count_outliers(df, 'Col2')} outliers")

# %% 
# %% [markdown]
# ## Resolve the result
#
# We can either accept this slightly strange looking result or we can keep on trimming outliers until 
# there are non left in the updated data frame ..

# %% Keep on removing outliers until they are all gone ...
for column in df:
    loop_count = 0
    outlier_count = count_outliers(df, column)
    while outlier_count > 0:
        df = remove_outliers(df, column)
        outlier_count = count_outliers(df, column)
        loop_count += 1
        if (loop_count > 100):
            break

    print(f"{column} has {count_outliers(df, column)} outliers")
    box_and_whisker(df, column)

# %% [markdown]
# ## Conclusion
#
# We have generated some test data with outliers to explore the probem space and then built some
# helper functions to resolve them.
# 
# Having removed the outliers using the inter quartile ranges we noted that there appeared to be some
# left near the tails and explained this result as follow; the remaining outliers were not quite outliers
# in the original data but when we modified the data by trimming the original outliers the boundaries changed.
#
# This is an odd looking result. We can either accept it, as the point of removing outliers would usually be
# to remove outlandish values skewing the results or the models the results will be used in, or we can simply 
# iterate around the data removing outliers until they are all gone. Two iterations usually suffices but the 
# function provided keeps going until they are all gone

# %%
