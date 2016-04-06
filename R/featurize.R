if (file.exists("R")) setwd("R")
source("function_library.R")

library(RSQLite)

# Connect to SQLite file database.
db = dbConnect(drv=RSQLite::SQLite(), dbname="inbound/sample.sqlite")

# Load all data into a dataframe.
raw_data = dbGetQuery(conn=db, statement="SELECT * FROM may2015_sample")
dim(raw_data)
names(raw_data)

save(raw_data, file="data/raw-data-sample.RData")

# Create features.
data = featurize(raw_data)

# Save an Rdata frame.
save(data, file="data/data-featurized.RData")

# Clean up.
rm(raw_data)
gc()