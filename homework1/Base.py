import json
import re
from sklearn import metrics


class Base:
    def __init__(self) -> None:
        self.model_loaded = False
        self.model = None
        self.model_path = None

    def load_model(self, model_path: str, model: any) -> None:
        '''Use this function to load model files'''
        self.model = model
        self.model_loaded = True
        self.model_path = model_path

    @staticmethod
    def replace_urls(data: str):
        return re.sub('https?\s?:\s?/\s?/\s?(?:[-\w.]\s?|(?:%[da-fA-F]{2}\s?))+', " urldata ",  data.replace('\n', '\n '))

    def pre_process(self, data: tuple):
        if not self.model_loaded:
            raise 'Model Not Loaded'

    def classify(self, data: tuple):
        pass

    def fit(self, x_train: list, y_train: list):
        pass

    def post_process(self, data: tuple):
        pass

    def train_data(self, data: tuple):
        '''This method should use pre-process function'''

    def process(self, data: tuple):
        return self.post_process(self.classify(self.pre_process(data)))

    @classmethod
    def kLapSmoothing(cls, prob, tot):
        return (prob + 1) / float(tot)

    @classmethod
    def clean_word(cls, word: str):
        if word == 'urldata':
            return '<url>'
        # This increases false Positives
        # elif word.isnumeric():
        #     return '<num>'
        # elif word == '-':
        #     return '-'
        # elif word not in stopwords.words('english'):
        #     return word
        elif '!' in word:
            return '!'
        elif word in ['.', '?']:
            return word
        elif word.isalpha():
            return word
        return None

    @classmethod
    def store_all(cls, var, path):
        with open(path, 'w') as f:
            f.seek(0)
            json.dump(var, f)

    def test_result(self, Y_actual: list, Y_predicted: list) -> dict:
        # confusion_matrix = metrics.confusion_matrix(Y_actual, Y_predicted)
        # cm_display = metrics.ConfusionMatrixDisplay(
        #     confusion_matrix=confusion_matrix, display_labels=self.model['classes'].values())
        # cm_display.plot()
        # plt.show()
        return {
            "recall_score": metrics.recall_score(Y_actual, Y_predicted),
            "precision_score": metrics.precision_score(Y_actual, Y_predicted),
            "accuracy_score": metrics.accuracy_score(Y_actual, Y_predicted),
            "f1_score": metrics.f1_score(Y_actual, Y_predicted)
        }
