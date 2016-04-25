import pandas as pd
import numpy as np

data = pd.read_csv('~/yelp_project_/test_photo_to_biz.csv')

data.sort_values("photo_id", inplace=True)

data.drop_duplicates("photo_id", inplace=True)

pseudo = '\t'.join(["0"]*9)

with open("test_labels.txt", "w+") as f:
    for index, row in data.iterrows():
        f.write(str(row["photo_id"]))
        f.write("\t")
        f.write(pseudo)
        f.write("\t")
        f.write(str(row["photo_id"]) + ".jpg" + "\n")

