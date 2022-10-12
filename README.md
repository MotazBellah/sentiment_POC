# Sentiment Analysis POC

REST API application performing a sentiment analysis to comment. In this POC we use three ways to do sentiment analysis.

- [VaderSentiment](https://github.com/cjhutto/vaderSentiment)
- [Pretrained model from hugging face](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Tweet NLP](https://tweetnlp.org/resources/)


## Clone/Run app
````
# Clone repo
$ git clone https://github.com/MotazBellah/sentiment_POC.git

# Go to dir
$ cd sentiment_POC

# Run
$ docker-compose up

````
After Starting the application will start to download the model.

## Testing
- Testing the weetnlp lib
````
curl --location --request POST 'http://localhost:8000/weetnlp' \
--header 'Content-Type: application/json' \
--data-raw '{
    "comment": "good"  
}'

````

- Testing the VaderSentiment lib
````
curl --location --request POST 'http://localhost:8000/vader' \
--header 'Content-Type: application/json' \
--data-raw '{
    "comment": "good"
}'

````

- Testing the hugging face model
````
curl --location --request POST 'http://localhost:8000/hugging_face' \
--header 'Content-Type: application/json' \
--data-raw '{
    "comment": "good"  
}'

````

