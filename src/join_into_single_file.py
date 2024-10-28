import pandas as pd
from pathlib import Path

processed_data_dir = Path("../data/processed_data")

# Load all files into a single dataframe, instead of user and session columns make single sess_id column with a combination of user and session

df_lst = []
for directory in processed_data_dir.iterdir():
    for file in directory.iterdir():
        activity_df = pd.read_csv(file)
        # session = activity_df["session"].unique()[0]
        # user = activity_df["user"].unique()[0]
        # activity_df["sess_id"] = int(str(user) + str(session))
        # activity_df.drop(columns=["user", "session"], inplace=True)
        df_lst.append(activity_df)

df = pd.concat(df_lst)

df.to_csv("remote_ctrl_new_train.csv", index=False)
