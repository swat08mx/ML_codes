from scipy import stats
import numpy as np

sample_A = np.array([78, 84, 92, 88, 75, 80, 85, 90, 87, 7978, 84, 92, 88, 75, 80, 85, 90, 87, 79])
sample_B = np.array([82, 88, 75, 90, 78, 85, 88, 77, 92, 8082, 88, 75, 90, 78, 85, 88, 77, 92, 80])

t_stat, p_val = stats.ttest_ind(sample_A, sample_B)
alpha = 0.05
df = len(sample_A)+len(sample_B)-2
critical_t = stats.t.ppf(1-alpha/2, df)

print("T_value:", t_stat)
print("P_value: ", p_val)
print("Critical t-value: ", critical_t)

print("With T-value")
if np.abs(t_stat) >  critical_t:
    print("There is significant difference between two groups")
else:
    print("There is no significant difference between two groups")

print("With P-value")
if p_val >alpha:
    print('No evidence to reject the null hypothesis that a significant difference between the two groups')
else:
    print('Evidence found to reject the null hypothesis that a significant difference between the two groups')