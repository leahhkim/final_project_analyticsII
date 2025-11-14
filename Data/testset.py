import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/newborn_health_monitoring_with_risk.csv")

train_val, test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

test.to_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/test_set.csv", index=False)
train_val.to_csv("/Users/leahkim/Desktop/final_project_analyticsII/Data/train_val.csv", index=False)
