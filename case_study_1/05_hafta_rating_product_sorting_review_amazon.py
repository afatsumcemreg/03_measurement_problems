####################################################
# project: rating product and sorting reviews in amazon
####################################################

# importing libraries
import pandas as pd
import math
from scipy import stats as st
from helpers import eda

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.expand_frame_repr', False)

# reading the dataset
df = pd.read_csv('C:/Users/test/PycharmProjects/miuul_data_sicence_bootcamp/datasets/amazon_review.csv')
df.head()
df.shape

# business problem
"""
One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales.
The solution to this problem means providing greater customer satisfaction for the e-commerce site, prominence of the product for the sellers and a seamless shopping experience for the buyers.
Another problem is the correct ordering of the comments given to the products.
Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers.
In the solution of these 2 basic problems, e-commerce site and sellers will increase their sales, while customers will complete their purchasing journey without any problems.
"""

# dataset story
"""
This dataset, which includes Amazon product data, includes product categories and various metadata.
The product with the most reviews in the electronics category has user ratings and reviews.

reviewerID:     User ID
asin:           Product ID
reviewerName:   Username
helpful:        Helpful rating rating
reviewText:     Review
overall:        Product rating
summary:        Evaluation summary
unixReviewTime: Evaluation time
reviewTime:     Reviewtime Raw
day_diff:       Number of days since evaluation
helpful_yes:    The number of times the review was found helpful
total_vote:     Number of votes given to the review
"""
# examining the dataset
eda.check_df(df)                                            # general picture
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)    # grabing categorical, numerical and cardinal variables
for col in cat_cols:                                        # summary of categorical variables
    print(col, eda.cat_summary(df, col))

for col in num_cols:                                        # summary of numerical variables
    eda.num_summary(df, col)

# Task 1: Calculate the Average Rating according to the current comments and compare it with the existing average rating
# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the scores given by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.

# Step 1: Calculate the average score of the product.
# overall distribution
df['overall'].mean()  # 4.587589013224822

# Step 2: Calculate the weighted average score by date.
# You need to declare the reviewTime variable as a date variable.
# Assume the max value of reviewTime as current_date
# you need to create a new variable by expressing the difference of each score-interpretation date
# and current_date in days, and divide the variable expressed in days by 4 with the quantile function
# (if 3 quarters are given, it will be 4 parts) and weight according to the values from the quartiles.
# For example, if q1 = 12, when weighting, averaging comments made less than 12 days ago and giving them a higher weight.

df['reviewTime'] = pd.to_datetime(df['reviewTime'])         # datetime64[ns]
current_date = df['reviewTime'].max()
df['days'] = (current_date - df['reviewTime']).dt.days
pd.qcut(df['days'], 4).value_counts()

def time_based_weighted_score(dataframe, w1=0.28, w2=0.26, w3=0.24, w4=0.22):
    return dataframe.loc[dataframe['days'] <= 280, 'overall'].mean() * w1 + dataframe.loc[
        (dataframe['days'] > 280) & (dataframe['days'] <= 430), 'overall'].mean() * w2 + dataframe.loc[
               (dataframe['days'] > 430) & (dataframe['days'] <= 600), 'overall'].mean() * w3 + dataframe.loc[
               (dataframe['days'] > 600), 'overall'].mean() * w4


time_based_weighted_score(df)                                       # 4.595593165128118
time_based_weighted_score(df, w1=0.40, w2=0.30, w3=0.20, w4=0.10)   # 4.628116998159475

# Step 3: Compare and interpret the average of each time period in weighted scoring.
df.loc[df['days'] <= 280, 'overall'].mean()                         # 4.6957928802588995
df.loc[(df['days'] > 280) & (df['days'] <= 430), 'overall'].mean()  # 4.636140637775961
df.loc[(df['days'] > 430) & (df['days'] <= 600), 'overall'].mean()  # 4.571661237785016
df.loc[(df['days'] > 600), 'overall'].mean()                        # 4.4462540716612375

# Task 2: Determine 20 reviews for the product to be displayed on the product detail page.

# Step 1: Generate the helpful_no variable.
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.
# Find the number of votes that are not helpful (helpful_no) by subtracting the number of helpful
# votes (helpful_yes) from the total number of votes (total_vote).
df['helpful_no'] = df['total_vote'] - df['helpful_yes']
df.head()

# Step 2: Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data.
# To calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores,
# define the score_pos_neg_diff, score_average_rating and wilson_lower_bound functions.
# Create scores based on score_pos_neg_diff. Next; Save it as score_pos_neg_diff in df.
# Create scores according to score_average_rating. Next; Save it as score_average_rating in df.
# Create scores according to wilson_lower_bound. Next; Save it as wilson_lower_bound in df.
def score_pos_neg_diff(up, down):
    return up - down

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df['score_pos_nef_diff'] = df.apply(lambda x: score_pos_neg_diff(x['helpful_yes'], x['helpful_no']), axis=1)
df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1)
df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)
df.head()

# Step 3: Identify the 20 Interpretations and Interpret the results.
# Identify and rank the top 20 comments according to wilson_lower_bound.
# Comment the results.
df.sort_values('wilson_lower_bound', ascending=False).head(20)
df.sort_values('score_pos_nef_diff', ascending=False).head(20)
df.sort_values('score_average_rating', ascending=False).head(20)