import pandas as pd
import seaborn as sns
from statsmodels.stats.multicomp import MultiComparison
import scikit_posthocs as sp
from scipy.stats import shapiro

df = sns.load_dataset('tips')
df.head()

# tukey test
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# duncan multiple comparison test
pvalue = sp.posthoc_dunn([df.loc[df['day'] == 'Thur', 'total_bill'],
    df.loc[df['day'] == 'Fri', 'total_bill'],
    df.loc[df['day'] == 'Sat', 'total_bill'],
    df.loc[df['day'] == 'Sun', 'total_bill']], p_adjust = 'holm')

print(pvalue)
pvalue > 0.05

pvalue = shapiro(df['total_bill'])[1]