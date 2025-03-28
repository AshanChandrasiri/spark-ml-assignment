import os
import traceback

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from threading import Timer
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, lower, col, udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.ml.feature import Word2VecModel
from pyspark.sql.types import ArrayType, StringType
import nltk
from nltk.stem import PorterStemmer
from pyspark.ml.classification import LogisticRegressionModel
from xgboost.spark import SparkXGBClassifierModel
# from pyngrok import ngrok
# import threading
import webbrowser

import pyspark

print(pyspark.__version__)

app = Flask(__name__)

BASE_MODEL_PATH = "./models"

XG_BOOST_MODEL_PATH = os.path.abspath(os.path.join(BASE_MODEL_PATH, "xg-boost"))
LR_MODEL_PATH = os.path.abspath(os.path.join(BASE_MODEL_PATH, "logistic_regression"))
WORD2VEC_MODEL_PATH = os.path.abspath(os.path.join(BASE_MODEL_PATH, 'word2vec'))

spark = (SparkSession.builder.appName("MusicClassification-app")
         .config("spark.hadoop.io.native.lib", "false")
         .config("spark.hadoop.io.nativeio", "false")
         .getOrCreate())

# print(lr_model_path)
lr_model = LogisticRegressionModel.load(LR_MODEL_PATH)
xg_boost_model = SparkXGBClassifierModel.load(XG_BOOST_MODEL_PATH)
word_2_vec_model = Word2VecModel.load(WORD2VEC_MODEL_PATH)

tokenizer = Tokenizer(inputCol="clean_lyrics", outputCol="words")
stop_words_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
nltk.download("punkt")
stemmer = PorterStemmer()

label_map = {
    0: 'pop',
    1: 'country',
    2: 'blues',
    3: 'rock',
    4: 'jazz',
    5: 'reggae',
    6: 'hip hop',
    7: 'soul'
}

port = 5000


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.json
        model_opt = data.get("model")
        lyrics = data.get("lyrics")

        if not model_opt or not lyrics:
            return jsonify({"error": "Missing model or lyrics"}), 400

        if model_opt == 'XGBOOST':
            model = xg_boost_model
        else:
            model = lr_model

        df = spark.createDataFrame([(lyrics,)], ["lyrics"])
        df = df.withColumn("clean_lyrics", lower(col("lyrics")))
        df = df.withColumn("clean_lyrics", regexp_replace(col("clean_lyrics"), "[^a-zA-Z\\s]", ""))

        df = tokenizer.transform(df)

        df = stop_words_remover.transform(df)

        stem_udf = udf(lambda words: [stemmer.stem(word) for word in words], ArrayType(StringType()))
        df = df.withColumn("stemmed_words", stem_udf(col("filtered_words")))

        df = word_2_vec_model.transform(df)

        y_pred = model.transform(df)
        predicted_genre = label_map[int(y_pred.collect()[0]['prediction'])]
        prediction_probs = y_pred.collect()[0]["probability"][int(y_pred.collect()[0]['prediction'])]

        # pred_results = {
        #     'pop': 0.65,
        #     'country': 0.15,
        #     'blues': 0.1,
        #     'rock': 0.01,
        #     'jazz': 0.02,
        #     'reggae': 0.05,
        #     'hip hop': 0.02
        # }

        probabilities = y_pred.collect()[0]['probability'].toArray()
        pred_results = {label_map[i]: float(probabilities[i]) for i in range(len(probabilities))}

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
            "prediction_label": predicted_genre,
            "prediction_score": prediction_probs,
            "bar_chart": f"data:image/png;base64,{bar_image_base64}",
            "pie_chart": f"data:image/png;base64,{pie_image_base64}"
        }

        return jsonify(resp)
    except Exception as e:
        print(traceback.format_exc())
        raise e


def start_app():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == '__main__':
    # public_url = ngrok.connect(port).public_url
    # print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    # app.config["BASE_URL"] = public_url
    # threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
    Timer(1, start_app).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
