if (file.exists("R")) setwd("R")
source("function_library.R")
load_libraries()

# Import data.
load("data/data-featurized.RData")

# Start h2o.
library(h2o)  # This will load the `h2o` R package as well

# Start an H2O cluster with nthreads = num cores on your machine
# TODO: support multi-node parallel cluster ala Savio.
h2o.init(nthreads=detectCores())

# Clean slate - just in case the cluster was already running
h2o.removeAll()

data = data_processed$data
dim(data)

# Remove comment content because it contains carriage returns and h2o can't handle it.
data$red_body = NULL

# Specify the target column.
y = "red_score"

# Remove the observations with missing values in our target variable.
data = data[!is.na(data[, y]), ]
nrow(data)
summary(data[, y], useNA="ifany")

# Convert to a 1/0 indicator.
data$red_score = as.numeric(data$red_score >= 2)
table(data$red_score, useNA="ifany")

# Save a backup of our R dataframe.
r_data = data
# Load data into h2o.
data = as.h2o(data)
# This is showing too many rows, but the correct number of columns. What's the deal?
# TODO: figure this out.
dim(data)
# head(data)
summary(data[, y])

data[, y] = as.factor(data[, y])
summary(data[, y], exact_quantiles=T)

# Divide into training and holdout.
# TODO: fix this placeholder and actually divide up the dataframes.
splits = h2o.splitFrame(
  data,         ##  splitting the H2O frame we read above
  c(0.7),   ##  create splits of 70% and 30%;
  seed=1234)    ##  setting a seed will ensure reproducible results (not R's seed)
# If we used splits of less than 100%, h2o would allocate a third split to the remainder.

train = h2o.assign(splits[[1]], "train.hex")
## assign the first result the R variable train
## and the H2O name train.hex
#valid = h2o.assign(splits[[2]], "valid.hex")   ## R valid, H2O valid.hex
holdout = h2o.assign(splits[[2]], "holdout.hex")     ## R test, H2O test.hex
# Double-check dimensions.
dim(train)
dim(holdout)

# Define parameters.



# Specify the names of our predictors, removing our target variable.
# Skip the first 22 columns, which we haven't processed yet.
features = names(data)[23:ncol(data)]

# Make sure that the target variable is not in the list of features.
x = setdiff(features, c(y))

length(x)

# Change to bernoulli if doing classification.
#distribution = "gaussian"
distribution = "bernoulli"

# Fit models.

gbm = h2o.gbm(x = x, y = y, training_frame = train, balance_classes=T, learn_rate=0.2, max_depth=5, ntrees=500)

h2o.saveModel(gbm, "model")

h2o.shutdown(prompt=F)

