import pandas as pd
import numpy as np

train_csv = pd.read_csv('/home/chris/yelp_project/train.csv')
photo2biz = pd.read_csv('/home/chris/yelp_project/train_photo_to_biz_ids.csv')

# merge two data sets
data = train_csv.merge(photo2biz, on='business_id')
data.sort_values("photo_id", inplace=True)
print data.columns

# 1-of-K way:


def onehot(value):
    try:
        foo = map(int, value.strip().split(" "))
        onehot = ["1" if i in foo else '0' for i in range(9)]
    except:
        onehot = ["0"] * 9
    return '\t'.join(onehot)

# print onehot(train_csv.iloc[0, 0], train_csv.iloc[0, 1])
with open("train_labels.txt", "w+") as f:
    for index, row in data.iterrows():
        f.write(str(row["photo_id"]))
        f.write("\t")
        f.write(onehot(row["labels"]))
        f.write("\t")
        f.write(str(row["photo_id"]) + ".jpg" + "\n")

