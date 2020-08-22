import numpy as np
import pandas as pd
from scipy import stats

# contingency table
#        click       no click
#------------------------------
# ad A |   A_click_Y      A_click_N
# ad B |   B_click_Y      B_click_N

data = pd.read_csv('advertisement_clicks.csv')
A_click_Y = data[(data.advertisement_id == 'A')&
                 (data.action == 1)].count()['advertisement_id']
A_click_N = data[(data.advertisement_id == 'A')&
                 (data.action == 0)].count()['advertisement_id']
B_click_Y = data[(data.advertisement_id == 'B')&
                (data.action == 1)].count()['advertisement_id']
B_click_N = data[(data.advertisement_id == 'B') &
                 (data.action == 0)].count()['advertisement_id']

T = np.array([[A_click_Y, A_click_N],[B_click_Y, B_click_N]])
chi2 = np.linalg.det(T)**2 * T.sum() / (T[0].sum()*T[1].sum()*T[:,0].sum()*T[:,1].sum())
p_value = 1-stats.chi2.cdf(x=chi2, df=1)

print(p_value)