from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder.appName('Recommender System CF').getOrCreate()
rat = spark.read.csv('datasets/rat.csv', inferSchema=True, header=True)
rat.printSchema()
data = rat.drop('timestamp')
data.show()
data.select(('rating')).describe().show()
(training, test) = data.randomSplit([0.8, 0.2])
als = ALS(maxIter=5,regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')
model = als.fit(training)
predictions = model.transform(test)
predictions.show() #predictions user to rate movie
mov = spark.read.csv('datasets/mov.csv', header=True, inferSchema=True)
res=predictions.join(mov, on=['movieId'], how='left')
res.show()
user_126 = res.filter(res['userId']==126).select(['userId', 'title', 'genres']) #get recommendation for user 126
user_564 = res.filter(res['userId']==564).select(['userId', 'title', 'genres']) #get recommendation for user 564
print('Recommendation Film for User 126')
user_126.show()
print('Recommendation Film for User 564')
user_564.show()
user_126.collect()
user_564.collect()
