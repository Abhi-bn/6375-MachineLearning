from BagOfWords import BagOfWords
import json


class Bernoulli(BagOfWords):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.model_path = 'Bernoulli.json'
        # self.word_dictionary = WordDictionary()
        self.load_model(path + "/" + self.model_path, {})

    def load_model(self, model_path: str, model: dict) -> None:
        try:
            with open(model_path, 'r') as f:
                super().load_model(model_path, json.load(f))
        except Exception as e:
            super().load_model(model_path, model)

    def pre_process(self, data: tuple):
        return BagOfWords.get_words(data, False)


# if __name__ == "__main__":
#     bofw = Bernoulli('train_data')
#     # bofw.train_data(
#     #     {'ham': 'enron1/train/ham', 'spam': 'enron1/train/spam'})

#     # bofw.test_data(
#     #     {'ham': 'enron1/test/ham', 'spam': 'enron1/test/spam'})

#     # bofw.train_data(
#     #     {'ham': 'enron4/train/ham', 'spam': 'enron4/train/spam'})

#     # bofw.test_data(
#     #     {'ham': 'enron4/test/ham', 'spam': 'enron4/test/spam'})

#     bofw.train_data(
#         {'ham': 'hw1/train/ham', 'spam': 'hw1/train/spam'})

#     bofw.test_data(
#         {'ham': 'hw1/test/ham', 'spam': 'hw1/test/spam'})
