

import json
import re
import pickle
from word_dictionary import WordDictionary


class BagOfWords:
    train_path = "train/ham/*"
    test_path = "train/spam/*"
    spam_data = "spam_data.json"
    ham_data = "ham_data.json"

    def __init__(self) -> None:
        from glob import glob
        self.train_data_path = glob(BagOfWords.train_path)
        self.test_data_path = glob(BagOfWords.test_path)
        self.store_train_of_words = {}
        self.store_test_of_words = {}
        self.word_dictionary = WordDictionary()

    def make_matrix(self):
        get_all_words = self.word_dictionary.get_uniq_words()
        Y = []
        X = [0] * len(get_all_words)
        for hash, words in self.store_train_of_words.items():
            for w, c in words.items():
                index = get_all_words.index(w)
                X[index] += c
            Y.append(1)
        with open('train_data/ham_matrix.txt', 'w') as f:
            json.dump(X, f)
        X = [0] * len(get_all_words)
        for hash, words in self.store_test_of_words.items():
            for w, c in words.items():
                index = get_all_words.index(w)
                X[index] += c
            Y.append(0)
        with open('train_data/spam_matrix.txt', 'w') as f:
            json.dump(X, f)
        self.word_dictionary.save()

    @classmethod
    def load_file_json(cls, filename: str):
        with open(filename, 'rb') as f:
            try:
                return pickle.load(f)
            except Exception as e:
                return set()

    @staticmethod
    def clean_data(data: str, period=False):
        reg = r"[^a-zA-Z]+"
        if period:
            reg = r"[^a-zA-Z.]+"
        return re.sub(reg, ' ', data)

    @staticmethod
    def store_all(var, path):
        with open(path, 'w') as f:
            json.dump(var, f)

    def store_unique(self, data: str) -> dict:
        from nltk.tokenize import sent_tokenize, word_tokenize

        lowercase = data.lower()
        tok = sent_tokenize(lowercase)
        clean = [[k for k in word_tokenize(
            t) if k.isalpha() or k == '.'] for t in tok]
        all_words = sum(clean, [])
        uiq = set(all_words)
        self.word_dictionary.add_to_dict(uiq)
        return {k: all_words.count(k) for k in uiq}

    def iterate_data(self) -> None:
        from hashlib import md5

        for path in self.train_data_path:
            with open(path, 'r', errors='ignore') as f:
                info = f.read()
                self.store_train_of_words[md5(
                    path.encode()).hexdigest()] = self.store_unique(info)
        BagOfWords.store_all(self.store_train_of_words, BagOfWords.ham_data)

        for path in self.test_data_path:
            with open(path, 'r', errors='ignore') as f:
                info = f.read()
                self.store_test_of_words[md5(
                    path.encode()).hexdigest()] = self.store_unique(info)
        BagOfWords.store_all(self.store_test_of_words, BagOfWords.spam_data)
        self.make_matrix()


if __name__ == "__main__":
    # save_unique_words()
    BagOfWords().iterate_data()
