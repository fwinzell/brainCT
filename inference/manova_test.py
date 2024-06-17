import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro
import numpy as np

# Sample data with multiple observations for each method and class
data = {
    'method': ['A']*9 + ['B']*9,
    'class': ['WM', 'WM', 'WM', 'GM', 'GM', 'GM', 'CSF', 'CSF', 'CSF']*2,
    'dice': [0.85, 0.86, 0.84, 0.87, 0.88, 0.86, 0.86, 0.85, 0.87, 0.83, 0.82, 0.84, 0.84, 0.83, 0.82, 0.82, 0.81, 0.83],
    'iou': [0.80, 0.81, 0.79, 0.82, 0.83, 0.81, 0.81, 0.80, 0.82, 0.78, 0.77, 0.79, 0.79, 0.78, 0.77, 0.77, 0.76, 0.78]
}

df = pd.DataFrame(data)

print(df.columns)
print(df.dtypes)

# Ensure 'method' and 'class' are categorical
df['method'] = pd.Categorical(df['method'])
df['class'] = pd.Categorical(df['class'])

print(df.dtypes)


# Checking for normality (Shapiro-Wilk test)
print("Shapiro test for Dice scores:", shapiro(df['dice']))
print("Shapiro test for IoU scores:", shapiro(df['iou']))

# Performing MANOVA
maov = MANOVA.from_formula('dice + iou ~ method * class', data=df)
print(maov.mv_test())