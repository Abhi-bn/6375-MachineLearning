from collections import OrderedDict
import json


class WordDictionary:
    path = 'unique_words.txt'

    def __init__(self) -> None:
        try:
            with open(WordDictionary.path, 'r') as f:
                self.unique_words = json.load(f)
        except Exception as e:
            self.unique_words = OrderedDict()

    def get_uniq_words(self):
        return list(self.unique_words)

    def add_to_dict(self, word):
        if type(word) == str:
            self.unique_words[word] = ''
            return
        for w in word:
            self.unique_words[w] = ''

    def save(self):
        with open(WordDictionary.path, 'w') as f:
            json.dump(self.unique_words, f)

    def has_word(self, word: str) -> bool:
        return word in self.unique_words

    # def __del__(self):
    #     self.save()
