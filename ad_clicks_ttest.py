import numpy as np
from scipy import stats
import pandas as pd

# generate data
data = pd.read_csv('advertisement_clicks.csv')

a = data[data.advertisement_id == 'A'].action
b = data[data.advertisement_id == 'B'].action

N = len(a)

# calc mean and var
mean_a = a.mean()
mean_b = b.mean()
var_a = a.var(ddof=1) # denominator = n-ddof
var_b = b.var(ddof=1)

s = np.sqrt((var_a + var_b) / 2)

# calc t stats
t = (mean_a - mean_b) / (s * np.sqrt(2 / N))

# degree of freedom
df = 2 * N - 2
p = 1 - stats.t.cdf(np.abs(t),df=df) # one-sided p value

print('t stats:',t,'p value:', p*2)

# Fastest way to calc t
t = stats.ttest_ind(a,b)
print(t)



