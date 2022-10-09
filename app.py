import os
from flask import jsonify, request, Flask
from sentiment import SentimentAnalysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

model = SentimentAnalysis("nlptown/bert-base-multilingual-uncased-sentiment")
pipe = model.model_pipe()


@app.route("/api/v1/vader",  methods=['POST'])
def vader():
    try:
        args = request.get_json()
        sentence = args['comment']
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    return jsonify({"result": vs})


@app.route("/api/v1/sentiment_analysis",  methods=['POST'])
def sentiment_analysis():
    try:
        args = request.get_json()
        sentence = args['comment']
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)})

    sentiment_map = {"1 star": "Very Negative", "2 stars": "Negative", "3 stars": "Neutral",
                     "4 stars": "Positive", "5 stars": "Very Positive",
                    }

    sentiment_result = pipe(sentence)
    for i in sentiment_result:
        for k, v in i.items():
            print(k, v)
            if k == "label":
                i[k] = sentiment_map[v]

    return jsonify({"result": sentiment_result})

   
if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=PORT)