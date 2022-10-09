from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline)
# from transformers import TFAutoModelForSequenceClassification
# from transformers import AutoTokenizer, AutoConfig


class SentimentAnalysis:
    def __init__(self, model_name) -> None:
        self.model_name = model_name

    def model_pipe(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.save_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_name)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        return pipe
