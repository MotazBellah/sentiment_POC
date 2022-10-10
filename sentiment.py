
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TextClassificationPipeline)
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax


class SentimentAnalysis:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def model_pipe(self):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        model.save_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_name)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        return pipe, tokenizer, config, model

    def text_analysis(self, text, tokenizer, config, model):
        result = {}
        # tokenizer, config, model = self.model_pipe()[1:]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        result["sentiment"] = config.id2label[ranking[0]]
        result["scores"] = {}
        for i in range(scores.shape[0]):

            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            result["scores"][l] = np.round(float(s), 4)

        return result
