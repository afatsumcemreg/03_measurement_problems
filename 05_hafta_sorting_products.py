##############################################
# sorting products
##############################################

##############################################
# application: sorting courses
##############################################

##############################################
# importing libraries
##############################################

import math
import pandas as pd
import scipy.stats as st 
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.expand_frame_repr', False)

##############################################
# reading the dataset
##############################################

df = pd.read_csv('C:/Users/test/PycharmProjects/miuul_data_sicence_bootcamp/datasets/product_sorting.csv')
df.head()
df.shape

##############################################
# sorting by rating 
##############################################

df.sort_values('rating', ascending=False)


##############################################
# sorting by purchase count or comment count
##############################################

# sorting by purchase count
df.sort_values('purchase_count', ascending=False)

# sorting by comment count
df['comment_count'] = df['commment_count']
df.sort_values('comment_count', ascending=False)

##############################################
# sorting by rating, purchase count and comment count
##############################################

# applying standardization with MinMaxScaler
# considering those three factors together
# converting purchase count to the values between 1-5
df['purchase_count_scaled'] = MinMaxScaler(feature_range = (1, 5)).fit(df[['purchase_count']]).transform(df[['purchase_count']])

# converting commment_count to the valuee between 1-5
df['commment_count_scaled'] = MinMaxScaler(feature_range = (1, 5)).fit(df[['commment_count']]).transform(df[['commment_count']])
df['new_variables'] = MinMaxScaler(feature_range = (1, 5)).fit_transform(df[['commment_count']]) # bir diger donusturme metodu
# noinspection PyStatementEffect
df[['commment_count_scaled', 'new_variables']]

# get the descriptive statistics
df.describe().T

# the weighted average of these three variables
df['purchase_count_scaled'] * 0.26 + df['commment_count_scaled'] * 0.32 + df['rating'] * 0.42

# functionalization of all above processes
def weighted_sorting_score(dataframe, w1=0.32, w2=0.26, w3=0.42):
    return dataframe['purchase_count_scaled'] * w1 + dataframe['commment_count_scaled'] * w2 + dataframe['rating'] * w3

df['weighted_sorting_score'] = weighted_sorting_score(df)

# sorting with the new_variable
df.sort_values('weighted_sorting_score', ascending=False)

# removing the observations without 'veri bilimi'
df[df['course_name'].str.contains('Veri Bilimi')].sort_values('weighted_sorting_score', ascending=False)

##############################################
# Bayesian Average Rating Score
##############################################

# defining a Bayesian Average Rating function
def bayesian_average_rating(n, confidence=0.95):    # n = number of each star given
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['1_point', '2_point', '3_point', '4_point', '5_point']]), axis=1)
df.sort_values('weighted_sorting_score', ascending=False).head()
df.sort_values('bar_score', ascending=False).head() # the best right score

# selecting the courses in the 5. and 1. indexes
df[df['course_name'].index.isin([5, 1])].sort_values('bar_score', ascending=False)

##############################################
# hybrit sorting = bar score + other factors
##############################################

# defining a function regarding hybrid sorting
def hybrid_sorting_score(dataframe, bar_w=0.60, wss_w=0.40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[['1_point', '2_point', '3_point', '4_point', '5_point']]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w + wss_score * wss_w

df['hybrid_sorting_score'] = hybrid_sorting_score(df)
df.sort_values('hybrid_sorting_score', ascending=False)

# sorting according to the data including 'veri bilimi'
df[df['course_name'].str.contains('Veri Bilimi')].sort_values('hybrid_sorting_score', ascending=False)