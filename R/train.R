if (file.exists("R")) setwd("R")
source("function_library.R")
load_libraries()

# Import data.
load("data/data-featurized.RData")

# Start h2o.
library(h2oEnsemble)  # This will load the `h2o` R package as well

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

# GBM gridsearch.
ntrees_opt = c(100, 200, 500)
max_depth_opt = c(2, 3, 5)
learn_rate_opt = c(0.1, 0.2)

hyper_params = list('ntrees' = ntrees_opt,
                    'max_depth' = max_depth_opt,
                    'learn_rate' = learn_rate_opt)

grid_search = h2o.grid(algorithm = "gbm",
                       grid_id = "gbm_grid",
                       hyper_params = hyper_params,
                       distribution=distribution,
                       x = x, y = y,  training_frame = train)
#validation_frame = valid)
grid_search

# Sort by ascending MSE.
#perf_table = h2o.getGrid(grid_id = "gbm_grid", sort_by = "mse", decreasing = F)
perf_table = h2o.getGrid(grid_id = "gbm_grid", sort_by = "auc", decreasing = T)
print(perf_table)

# Extract the model with minimum CV-MSE.
best_model = h2o.getModel(perf_table@model_ids[[1]])
# Confirm the MSE.
#h2o.mse(best_model)
h2o.auc(best_model)

# Review variable importance.
print.data.frame(h2o.varimp(best_model)[1:30, ])

# What's the RMSE?
if (distribution == "gaussian") {
  sqrt(h2o.mse(best_model))
}

# Now that we can one model working, let's do SuperLearning.
# Following http://learn.h2o.ai/content/tutorials/ensembles-stacking/index.html

# Setup SuperLearner
# NOTE: glm gives an error, so remove for now. Perhaps due to NAs in the dataframe?
# Or may have been due to missing Y values, which has now been fixed.
# Error -- water.DException$DistributedException: from /127.0.0.1:54321; by class hex.ModelBuilder$1; class water.DException$DistributedException: from /127.0.0.1:54321; by class hex.glm.GLMTask$GLMGaussianGradientTask; class java.lang.ArrayIndexOutOfBoundsException: 429
#learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper",
#             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")

learner = c("h2o.randomForest.wrapper",
             "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner = "h2o.glm.wrapper"

# TODO: convert this to fitting the models separately, so that we can try different stackers.
fit_ens <- h2o.ensemble(x = x, y = y,  training_frame = train,
                    family = "AUTO",  learner = learner,
                    metalearner = metalearner, cvControl = list(V = 5))
# Review the results.
fit_ens$metafit

# Review performance
# perf = h2o.ensemble_performance(fit, newdata = data, score_base_models = F)
# print(perf)

# perf <- h2o.ensemble_performance(fit, newdata = test, score_base_models = FALSE)

# NNLS metalearner via https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble
h2o.glm_nn = function(..., non_negative = TRUE) {
  h2o.glm.wrapper(..., non_negative = non_negative)
}
metalearner = "h2o.glm_nn"

fit_ens2 = h2o.ensemble(x = x, y = y,  training_frame = train,
                     family = "AUTO",  learner = learner,
                     metalearner = metalearner, cvControl = list(V = 5))

# Review the model weights in the metalearner and training set performance.
fit_ens2$metafit

# Review performance on holdout (or external cross-validation).
# perf = h2o.ensemble_performance(fit, newdata = test, score_base_models = F)


# Shut down the h2o instance.
h2o.shutdown(prompt=F)
