# Reddit supervised learning (R implementation)

## 1. Checkout Github repository to local computer.

## 2. Setup data

* Download sample.sqlite and place into inbound/ folder.

## 3. Featurize - Run featurize.R

* Install required libraries as needed (see load_libraries()).
* Currently downsamples the sample data to 1%, to speed computation.
* See function_library.R for all of the processing functions.

## 4. Train - Run train.R

* This uses h2o for multicore parallelism.
