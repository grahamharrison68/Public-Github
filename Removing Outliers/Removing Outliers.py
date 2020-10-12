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
# ![](IQR.png)

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
# ![](IQR.png)

# %% [markdown]
# ### Helper functions
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
# Having done all the heavy lifting in the helper functions we can now go ahead and remove the rows from the outliers
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

# %% [markdown]
# ### Create a new helper function
#
# This helpder function will call remove_outliers repeatedly until all outliers have been removed
# including the ones that were inside the old boundaires but just outside of the new ones.

# %% New helper function ...
def remove_all_outliers(df_in, col_name):
    loop_count = 0
    outlier_count = count_outliers(df_in, col_name)

    while outlier_count > 0:
        loop_count += 1

        if (loop_count > 100):
            break

        df_in = remove_outliers(df_in, col_name)
        outlier_count = count_outliers(df_in, col_name)
    
    return df_in

# %% Keep on removing outliers until they are all gone ...
for column in df:
    df = remove_all_outliers(df, column)
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

# %% [markdown]
# ## Putting it into action
#
# Now we know how to remove outliers (including the odd looking result when new outliers have been 
# created when the minimum and maximum boundaires move) let's put it all into action by looking at
# a typical pattern that occurs in exploratory data analysis ...

# ### A normal distribution ...
#
# The histogram below is normally distribute and if we were given this data as part of a data science
# project we would be moving onto the next stage, looking for correlations etc. 

# %% Create a normal distriubtion with some outliers 
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000) # create a 1000 normally distributed data points

df_normal = pd.DataFrame({'Col0': s})
df_normal['Col0'].hist()

# %% [markdown]
# ### A skewed distribution ...
#
# However, if we force some of our normally distributed data to be extreme outliers then the 
# plotted distribution takes on a very different shape ...

# %% Add some outliers and re-plot 
s[600] = 6
s[700] = 6.5
s[800] = 6.57
s[900] = 6.8

df_normal = pd.DataFrame({'Col0': s})
df_normal['Col0'].hist()

# %% [markdown]
# ### Resolving the skewed distribution ...
#
# The new plot is a common pattern we often see when plotting a feature in a histogram. A single big
# bar with a tiny bar way out to the left or right (or both) is a tell-tale sign that outliers might
# be present in the data and this means that our nice, tidy, normally distributed histogram is completely 
# hidden and obscurred by the single big bar.
#
# When we observ this pattern we need to remove the outliers and then see what the new distribution looks like.
# If we want to check for the presence of outliers then a quick box plot will confirm or deny ...
# %% Plot the box-and-whisker
box_and_whisker(df_normal, 'Col0')

# %% [markdown]
# ### Resolving the skewed distribution ...
#
# Sure enough there are outliers well outside the maximum. Fortunately we now have some helper functions defined 
# that can do this for us with minimal effort.

# %% Remove the outliers and re-display the box-and-whisker and the histogram
df_normal = remove_all_outliers(df_normal, 'Col0')
box_and_whisker(df_normal, 'Col0')
df_normal['Col0'].hist()

# %% [markdown]
# ## Closing Notes
#
# Having understood how to spot and remove outliers (properly) we have also worked through spotting 
# the tell-tale sign of a single tall bar and a distant small one in a histogram that often indicates 
# the presence of outliers. 
#
# We have then applied the helper functions in this more realistic scenario and demonstrated that
# the outliers have been removed which has enabled a standard histogram to reveal the true pattern 
# of distribution within our data points.

# %%