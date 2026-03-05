import pandas as pd
from sklearn.metrics import accuracy_score
import os

ground_truth = pd.read_csv("../data/test_labels.csv")

leaderboard = []

for file in os.listdir("../submissions"):

    submission = pd.read_csv(f"../submissions/{file}")

    acc = accuracy_score(
        ground_truth["label"],
        submission["label"]
    )

    leaderboard.append([file,acc])

leaderboard = pd.DataFrame(
    leaderboard,
    columns=["team","accuracy"]
)

leaderboard = leaderboard.sort_values(
    by="accuracy",
    ascending=False
)

leaderboard.to_csv("../leaderboard/leaderboard.csv",index=False)

print(leaderboard)