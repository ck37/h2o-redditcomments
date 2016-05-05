if (file.exists("R")) setwd("R")
source("function_library.R")
load_libraries()

# Import data.
load("data/data-featurized.RData")
library(h2oEnsemble)  # This will load the `h2o` R package as well

data = data_processed$data
dim(data)

cat("PRINTING: ", data[0,]$red_body)
