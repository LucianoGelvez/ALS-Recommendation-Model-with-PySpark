# ALS Recommendation Model with PySpark

This project implements an Alternating Least Squares (ALS) recommendation model using PySpark to predict user ratings for movies.

## Dataset

The dataset used in this project is called `movielens_ratings.csv` and contains movie ratings provided by users. The attributes included are:

*   **movieId:** ID of the movie.
*   **rating:** Rating given by the user.
*   **userId:** ID of the user.

## Libraries

The following Python libraries are used in the project:

*   `pyspark.sql`: For data manipulation with Spark SQL.
*   `pyspark.ml.recommendation`: For the ALS algorithm.
*   `pyspark.ml.evaluation`: For model evaluation using RegressionEvaluator.

## Process

1.  **Data Loading:** The `movielens_ratings.csv` dataset is loaded from a CSV file using Spark.
2.  **Data Preprocessing:**
    *   Splitting data into training and test sets.
3.  **Application of ALS Algorithm:**
    *   Creation of the ALS model specifying user, item and rating columns.
    *   Training of the model with the training data.
4.  **Evaluation:**
    *   Prediction of ratings in the test data set.
    *   Calculation of Root Mean Squared Error (RMSE) to evaluate the model's performance.
5.  **Recommendation:**
    *   Filtering of the test data for a specific user.
    *   Predicting the ratings of the test data for a specific user.
    *   Ordering by prediction value to get recommendations

## Usage

To run this code, make sure you have the following installed:

*   Python 3.6+
*   Apache Spark
*   PySpark
*   `movielens_ratings.csv` file in the same directory as the script or in an accessible path.

You can run the code directly in a Spark environment, using a Jupyter notebook or a Python script.

## Additional Considerations

*   The quality of the model relies heavily on the amount and diversity of training data.
*   The value of `maxIter` for the ALS model is an important parameter that will improve performance, more iterations will improve performance, but will take longer to train.
