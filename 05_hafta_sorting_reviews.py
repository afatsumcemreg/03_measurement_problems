##############################################
# sorting products
##############################################

# the most accurate 'social proof'

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
# up-down difference score
##############################################

# defing a function to calculate the up-down difference
def score_up_down_diff(up, down):
    return up - down

# review 1: up=600, down=400
# review 2: up=5500, down=4500
score_up_down_diff(600, 400)    # for review 1
score_up_down_diff(5500, 4500) # for review 2

##############################################
# average rating
##############################################

# average rating = up ratio
# average rating = up / (up + down) = up / all ratings
# returns the ratio of the positive values

# defining a function to calculate the average rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)  # for review 1
score_average_rating(5500, 4500) # for review 2

# review 3: up=2, down=0
# review 4: up=100, down=1

score_average_rating(2, 0) # for review 3
score_average_rating(100, 1) # for review 4
# in this function, the frequency height is not considered

##############################################
# wilson lower bound score (wlb)
##############################################

# defining the wilson lower bound function
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0: return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600, 400)    # for review 1
wilson_lower_bound(5500, 4500)  # for review 2
wilson_lower_bound(2, 0)        # for review 3
wilson_lower_bound(100, 1)      # for review 4

wilson_lower_bound(0, 100)    # for review 1
wilson_lower_bound(0, 0)  # for review 2
wilson_lower_bound(78, 36)        # for review 3
wilson_lower_bound(100, 1)      # for review 4

# example
up = [1115, 454, 258, 253, 220]
down = [43, 35, 26, 19, 9]

df = pd.DataFrame({'up': up, 'down': down})
df['score_up_down_diff'] = df.up - df.down
df['score_average_rating'] = df.up / (df.up + df.down)
wilson_score = [0.95036, 0.9028, 0.86924, 0.89349, 0.92701]
df['wilson_score'] = wilson_score

df.sort_values('wilson_score', ascending=False)
df.sort_values('score_up_down_diff', ascending=False)
df.sort_values('score_average_rating', ascending=False)