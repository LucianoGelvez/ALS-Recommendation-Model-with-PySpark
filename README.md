# K-Means Clustering with PySpark

This project implements the K-Means Clustering algorithm using PySpark to group data from different types of seeds.

## Dataset

The dataset used in this project is called `seeds_dataset.csv` and contains information about different types of seeds. The attributes included are:

*   **area:** Area of the seed
*   **perimeter:** Perimeter of the seed
*   **compactness:** Compactness of the seed
*   **length_of_kernel:** Length of the kernel
*   **width_of_kernel:** Width of the kernel
*   **asymmetry_coefficient:** Asymmetry coefficient
*   **length_of_groove:** Length of the groove

## Libraries

The following Python libraries are used in the project:

*   `pyspark.sql`: For data manipulation with Spark SQL.
*   `pyspark.ml.clustering`: For the K-Means algorithm.
*   `pyspark.ml.feature`: For feature engineering (VectorAssembler, StandardScaler).

## Process

1.  **Data Loading:** The `seeds_dataset.csv` dataset is loaded from a CSV file using Spark.
2.  **Data Preprocessing:**
    *   Selection of relevant columns for the model.
    *   Transformation of the selected columns into a single feature vector using `VectorAssembler`.
    *   Scaling of features using `StandardScaler` to improve the algorithm's performance.
3.  **Application of K-Means Algorithm:**
    *   Creation of the K-Means model with `k=3` (three clusters).
    *   Training of the model with preprocessed data.
4.  **Evaluation:**
    *   Calculation of Within Set Sum of Squared Errors (WSSSE) to evaluate the model.
    *   Display of cluster centers.
    *  Display of predictions

## Usage

To run this code, make sure you have the following installed:

*   Python 3.6+
*   Apache Spark
*   PySpark
*   `seeds_dataset.csv` file in the same directory as the script or in an accessible path.

You can run the code directly in a Spark environment, using a Jupyter notebook or a Python script.

## Additional Considerations

*   This project is an example of unsupervised learning, therefore there are no real labels to evaluate the results.
*   Scaling data is an important process to prevent low-dimensional models from being negatively affected.