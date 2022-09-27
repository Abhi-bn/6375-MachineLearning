import json
from Base import Base
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from glob import glob


class BagOfWords(Base):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.model_path = 'BagOfWords.json'
        self.load_model(path + "/" + self.model_path, {})

    def load_model(self, model_path: str, model: dict) -> None:
        try:
            with open(model_path, 'r') as f:
                super().load_model(model_path, json.load(f))
        except Exception as e:
            super().load_model(model_path, model)

    @staticmethod
    def get_words(data: tuple, count_word=True):
        data = Base.replace_urls(data.lower())
        tok = sent_tokenize(data)
        all_words = []
        for t in tok:
            for k in word_tokenize(t):
                c = BagOfWords.clean_word(k)
                if c:
                    all_words.append(c)

        uiq = set(all_words)
        return {k: all_words.count(k) if count_word else 1 for k in uiq}

    def pre_process(self, data: tuple):
        return BagOfWords.get_words(data)

    def train_data(self, data: dict):
        classes = []
        self.uniq_words = set()

        for index, key in enumerate(data):
            index = str(index)
            # set class names
            self.model.setdefault('classes', {})
            self.model['classes'][index] = key
            self.model.setdefault('class_to_index', {})
            self.model['class_to_index'][key] = str(index)
            classes.append(index)
            # set class <key>
            self.model.setdefault(index, {})

            this_class_data = glob(data[key] + "/*.*")
            self.model[index]['data_count'] = len(this_class_data)

            c_words = 0
            this_class_words = {}
            for path in this_class_data:
                with open(path, 'r', errors='ignore') as f:
                    for new_wd, c in self.pre_process(f.read()).items():
                        this_class_words.setdefault(new_wd, 0)
                        this_class_words[new_wd] += c
                        self.uniq_words.add(new_wd)
                        c_words += c

            self.model[index]['feature_vector'] = this_class_words
            self.model[index]['total_words'] = c_words

        self.uniq_words = sorted(self.uniq_words)

        for index, _ in enumerate(data):
            index = str(index)
            l = [0] * len(self.uniq_words)
            for k, v in self.model[index]['feature_vector'].items():
                i = self.uniq_words.index(k)
                l[i] = v
            self.model[index]['feature_vector'] = l

        self.model['unique_words'] = self.uniq_words
        self.model['unique_words_len'] = len(self.uniq_words)
        self.model['total_data'] = sum(
            self.model[index]['data_count'] for index in classes)

        # update all the info after getting all the unique words
        for index in classes:
            self.model[index]['prob'] = self.model[index]['data_count'] / \
                self.model['total_data']

        BagOfWords.store_all(self.model, self.model_path)

    def classify(self, data: tuple):
        cl = self.model['classes']
        results = {}
        for c in cl:
            values = []
            for d in data:
                try:
                    i = self.model['unique_words'].index(d)
                    values.append(self.model[c]['feature_vector'][i])
                except Exception as e:
                    # We need to consider data which is not there in dictionary for laplacian smoothing to do its magic
                    # values.append(0)
                    pass
            prob = np.array(values, dtype=np.float64)
            tot = self.model[c]['total_words'] + self.model['unique_words_len']
            prob = self.kLapSmoothing(prob, tot)
            # Adding class prob
            np.append(prob, self.model[c]['prob'])
            results[c] = np.sum(np.log(prob, where=prob > 0))

        super().classify(data)
        return results

    def post_process(self, data: tuple):
        super().post_process(data)
        m = max(data, key=data.get)
        return self.model['classes'][m], data[m], data

    def test_data(self, data):
        Y_actual = []
        Y_predicted = []
        for key, value in data.items():
            this_class_data = glob(data[key] + "/*.*")
            for p in this_class_data:
                with open(p, 'r', errors='ignore') as f:
                    c, confi, all = self.process(f.read())
                    Y_actual.append(int(self.model['class_to_index'][key]))
                    Y_predicted.append(int(self.model['class_to_index'][c]))

        return self.test_result(Y_actual, Y_predicted)


# if __name__ == "__main__":
#     # save_unique_words()
#     bofw = BagOfWords('train_data')

#     # bofw.train_data(
#     #     {'ham': 'enron1/train/ham', 'spam': 'enron1/train/spam'})

#     # bofw.test_data(
#     #     {'ham': 'enron1/test/ham', 'spam': 'enron1/test/spam'})

#     bofw.train_data(
#         {'ham': 'enron4/train/ham', 'spam': 'enron4/train/spam'})

#     bofw.test_data(
#         {'ham': 'enron4/test/ham', 'spam': 'enron4/test/spam'})

#     # bofw.train_data(
#     #     {'ham': 'hw1/train/ham', 'spam': 'hw1/train/spam'})

#     # bofw.test_data(
#     #     {'ham': 'hw1/test/ham', 'spam': 'hw1/test/spam'})
