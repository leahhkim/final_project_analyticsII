import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/newborn_health_monitoring_with_risk.csv")

BABY_ID = "baby_id"   # change if your column name differs

# optional safety
df.columns = df.columns.str.strip()

# 1) split the UNIQUE baby IDs
unique_babies = df[BABY_ID].unique()

train_babies, test_babies = train_test_split(
    unique_babies,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

# 2) build row-level datasets using those baby sets
train_val = df[df[BABY_ID].isin(train_babies)].copy()
test      = df[df[BABY_ID].isin(test_babies)].copy()

# 3) save
train_val.to_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/train_val.csv", index=False)
test.to_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/test_set.csv", index=False)