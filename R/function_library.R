load_libraries = function() {
  # Hide all the extra messages displayed when loading these libraries.
  suppressMessages({
    library(R.matlab)  # To load the Matlab data.
    library(ggplot2)   # For charts
    library(reshape2)  # For heatmaps
    library(e1071)     # For svm
    library(foreach)   # For multicore
    library(doMC)       # For multicore (not used at the moment)
    library(doParallel) # For multicore
    library(RhpcBLASctl) # Accurate physical core detection
    library(dplyr)     # For group_by
    library(tm)        # For text mining
    library(SnowballC) # For stemming
    library(NLP)       # For ngrams and sentence tokenization.
    library(openNLP)   # For sentence tokenization, part-of-speech tagging and named-entity recognition.
    library(stringi)   # For sentence feature engineering.
    library(stringr)   # For sentence feature engineering.
    library(caret)     # For stratified cross-validation folds.
    library(eqs2lavaan) #  For covariance heatmap.
    #library(qdap)      # Not sure if we actually need this one.
  })
}

featurize = function(data) {
  # TBD
}
