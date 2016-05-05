import h2o
import numpy as np
import random

h2o.init()

data = h2o.import_file(path='featurized_data')
y_labels = np.load('labels')
y_labels = (y_labels > 5) + 0 # as type essentially
y_labels = np.reshape(y_labels, (len(y_labels), 1))

data = data.cbind(h2o.H2OFrame(y_labels))

rand = data.runif()
train = data[r < 0.6]
valid = data[(r >= 0.6) & (r < 0.9)]
test = data[r >= 0.9]

y = data.col_names()[-1]
x = data.col_names()[:-1]

gbm = h2o.gbm(x = x, y = y, training_frame = train, validation_frame = valid, max_depth = 5, ntrees=500, learn_rate=0.2, distribution="bernoulli")

print gbm

r = int(random.random() * len(y_labels))
sample = data[r,:]
prediction = h2o.predict(gbm, sample)
print prediction

h2o.save_model(gbm, "model")


