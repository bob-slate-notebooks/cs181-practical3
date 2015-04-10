import numpy as np
import csv
import graphlab as gl

# Use collaborative filtering to make recommendations

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'

# Load the training data.
# train_data = {}
# with open(train_file, 'r') as train_fh:
#     train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
#     next(train_csv, None)
#     for row in train_csv:
#         user   = row[0]
#         artist = row[1]
#         plays  = int(row[2])
    
#         if not user in train_data:
#             train_data[user] = {}
        
#         train_data[user][artist] = plays

data = gl.SFrame.read_csv(train_file, column_type_hints={"plays":int})
model = gl.recommender.create(data, user_id="user", item_id="artist", target="plays")

# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            soln_csv.writerow([id, model.predict(gl.SFrame(users=user), gl.SFrame(items=artist))])