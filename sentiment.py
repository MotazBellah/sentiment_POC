
from transformers import (AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TextClassificationPipeline)
import numpy as np
from scipy.special import softmax


class SentimentAnalysis:
    """
    A class to represent a sentiment analysis model.
    
    Attributes:
        model_name (str): the model name from the hugging face hub
    """

    def __init__(self, model_name) -> None:
        """
        Constructs all the necessary attributes for the SentimentAnalysis object.

        Parameters:
            model_name (str): the model name from the hugging face hub
        """
        
        self.model_name = model_name


    def model_pipe(self):
        """
        The function to save the pretrained model.
  
        Parameters:
            None.
          
        Returns:
            four tuples : pipe, tokenizer, config, model.
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        config = AutoConfig.from_pretrained(self.model_name)

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        model.save_pretrained(self.model_name)
        tokenizer.save_pretrained(self.model_name)

        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        return pipe, tokenizer, config, model

    def text_analysis(self, text, tokenizer, config, model):
        """
        The function to do full sentiment analysis of the text.
  
        Parameters:
            text (str): text/comment to do analysis on it.
            tokenizer (object): Model tokenizer to split the sentences into smaller units
                            that can be more easily assigned meaning.
            config (object): Model configuration object.
            model (object): ML model.
          
        Returns:
            result (dict): the full result of the sentiment analysis in json.
        """
        result = {}
        result["scores"] = {}
        # tokenizer, config, model = self.model_pipe()[1:]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        result["sentiment"] = config.id2label[ranking[0]]
        
        for i in range(scores.shape[0]):

            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            result["scores"][l] = np.round(float(s), 4)

        return result
