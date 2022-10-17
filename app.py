import os
from flask import jsonify, request, Flask
from sentiment import SentimentAnalysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweetnlp
from dotenv import load_dotenv

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

print("""
    #####################################################
    Downloading pretrained model from hugging face hub...
    #####################################################
    """)
# TweetNLP model from cardiffnlp
DATA_MODEL = SentimentAnalysis("cardiffnlp/twitter-xlm-roberta-base-sentiment")
# multilingual model (Smaller one for testing)
# DATA_MODEL = SentimentAnalysis("nlptown/bert-base-multilingual-uncased-sentiment")
# Initiate the pipeline from the model
pipe, tokenizer, config, model = DATA_MODEL.model_pipe()

print("""
    ########################################
    Downloading tweet model from tweetnlp...
    ########################################
    """)
tweet_model = tweetnlp.load('sentiment_multilingual')


load_dotenv()

endpoint = os.getenv('azure_endpoint')
key = os.getenv('azure_key')


@app.route("/vader", methods=['POST'])
def vader():
    """
    vader() function do a sentiment analysis for the comment
    using vaderSentiment lib. it accept only post requests,
    and get the comment value from thr request body
    :return: json value of the sentiment analysis
    """
    try:
        args = request.get_json()
        sentence = args['comment']
    except Exception as e:
        return jsonify({"error": "No comment found!"})

    result = {}
    result["scores"] = {}

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)

    if vs["compound"] >= 0.05:
        result["sentiment"] = "Positive"
    elif vs["compound"] <= -0.05:
        result["sentiment"] = "Negative"
    else:
        result["sentiment"] = "Neutral"

    vs.pop("compound")
    result["scores"].update(vs)
    return jsonify({"result": result})


@app.route("/hugging_face", methods=['POST'])
def sentiment_analysis():
    """
    sentiment_analysis() function do a sentiment analysis for the comment
    using pretrained ML model. it accept only post requests,
    and get the comment value from thr request body
    :return: json value of the sentiment analysis
    """
    try:
        args = request.get_json()
        sentence = args['comment']
    except Exception as e:
        return jsonify({"error": "No comment found!"})

    total_sentiment = {}

    sentiment_result = DATA_MODEL.text_analysis(sentence, tokenizer, config, model)
    sentiment_azure = DATA_MODEL.azure(sentence, endpoint, key)

    total_sentiment["AI_Service"] = sentiment_result
    total_sentiment["Azure"] = sentiment_azure if "sentiment" in sentiment_azure else {}

    return jsonify({"result": total_sentiment})


@app.route("/tweetnlp", methods=['POST'])
def sentiment():
    """
    sentiment() function do a sentiment analysis for the comment
    using pretrained ML model from tweetnlp. it accept only post requests,
    and get the comment value from thr request body
    :return: json value of the sentiment analysis
    """
    try:
        args = request.get_json()
        sentence = args['comment']
    except Exception as e:
        return jsonify({"error": "No comment found!"})

    # tweet_model = tweetnlp.load('sentiment_multilingual')
    total_sentiment = {}

    sentiment_result = tweet_model.sentiment(sentence)
    sentiment_azure = DATA_MODEL.azure(sentence, endpoint, key)

    total_sentiment["AI_Service"] = sentiment_result
    total_sentiment["Azure"] = sentiment_azure if "sentiment" in sentiment_azure else {}

    return jsonify({"result": total_sentiment})


if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=PORT)