from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("ALSRecommendation").getOrCreate()

# Load the data
data = spark.read.csv('movielens_ratings.csv', inferSchema=True, header=True)

# Split the data into training and test sets
training, test = data.randomSplit([0.8, 0.2], seed=42)

# Configure the ALS model
als = ALS(maxIter=10, userCol='userId', itemCol='movieId', ratingCol='rating',regParam=0.05, rank = 15)

# Train the model
model = als.fit(training)

# Generate predictions
predictions = model.transform(test)

# Evaluate the model with RMSE
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

# Filter test data for a specific user (e.g., userId 11)
single_user = test.filter(test['userId'] == 11).select(['movieId', 'userId'])

# Generate recommendations for the user
recommendations = model.transform(single_user)

# Show recommendations
recommendations.orderBy('prediction', ascending=False).show()

spark.stop()