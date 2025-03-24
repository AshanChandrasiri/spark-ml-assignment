import os
import traceback

import matplotlib.pyplot as plt
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from pyspark.ml.tuning import CrossValidatorModel

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, lower, col, monotonically_increasing_id, explode, udf
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Word2VecModel
from pyspark.sql.types import ArrayType, StringType
import nltk
from nltk.stem import PorterStemmer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel

app = Flask(__name__)

model_base_path = ".\saved_model"
spark = (SparkSession.builder.appName("MusicClassification-app")
         # .config("spark.hadoop.io.nativeio.disabled", "true")
         .getOrCreate())

# lr_model = CrossValidatorModel.load("saved_model/logistic_regression-v2")


label_map = {
    1: 'pop',
    2: 'country',
    3: 'blues',
    4: 'rock',
    5: 'jazz',
    6: 'reggae',
    7: 'hip hop'
}


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:

        # print(os.path.exists(os.path.join(model_base_path, "logistic_regression-v2")))
        # lr_model_path = os.path.join(model_base_path, "logistic_regression-v2")
        # # lr_model_path = os.path.abspath(lr_model_path)
        #
        # if not os.path.exists(lr_model_path):
        #     raise FileNotFoundError(f"Model path not found: {lr_model_path}")
        #
        # print(lr_model_path)
        # lr_model = LogisticRegressionModel.load(lr_model_path)
        # word2Vec_model = Word2VecModel.load(os.path.join(model_base_path, 'word2vec-new'))
        #
        # data = request.json
        # model = data.get("model")
        # lyrics = data.get("lyrics")
        #
        # if not model or not lyrics:
        #     return jsonify({"error": "Missing model or lyrics"}), 400
        #
        # print(data)
        #
        # model = lr_model
        #
        # # Dummy response (Replace with actual model prediction logic)
        # prediction = f"Prediction using {model} for lyrics: {lyrics[:30]}..."  # Shortened preview
        #
        # spark = SparkSession.builder.appName("MusicClassification").getOrCreate()
        # df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        # df = df.withColumn("clean_lyrics", lower(col("lyrics")))
        # df = df.withColumn("clean_lyrics", regexp_replace(col("clean_lyrics"), "[^a-zA-Z\\s]", ""))
        #
        # tokenizer = Tokenizer(inputCol="clean_lyrics", outputCol="words")
        # df = tokenizer.transform(df)
        #
        # stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        # df = stop_words_remover.transform(df)
        #
        # nltk.download("punkt")
        # stemmer = PorterStemmer()
        # stem_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
        # df = df.withColumn("stemmed_words", stem_udf(col("filtered_words")))
        #
        # df = word2Vec_model.transform(df)
        #
        # y_pred = model.transform(df)
        # predicted_genre = label_map[int(y_pred.collect()[0]['prediction'])]
        # prediction_probs = y_pred.collect()[0]["probability"]
        #
        # print(f'Prediction result ---> {predicted_genre} {prediction_probs}')

        pred_results = {
            'pop': 0.65,
            'country': 0.15,
            'blues': 0.1,
            'rock': 0.01,
            'jazz': 0.02,
            'reggae': 0.05,
            'hip hop': 0.02
        }

        labels = list(pred_results.keys())
        sizes = list(pred_results.values())

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f'])
        plt.title("Genre Prediction Probabilities")
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        pie_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f'])
        plt.xlabel("Genres")
        plt.ylabel("Probability")
        plt.title("Genre Prediction Probabilities")
        plt.xticks(rotation=45)
        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        plt.close()
        buffer.seek(0)
        bar_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        resp = {
            "prediction_label": "Hip Pop",
            "prediction_score": 64,
            "bar_chart": f"data:image/png;base64,{bar_image_base64}",
            "pie_chart": f"data:image/png;base64,{pie_image_base64}"
        }

        print(resp)

        return jsonify(resp)
    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == '__main__':
    app.run(debug=True)
