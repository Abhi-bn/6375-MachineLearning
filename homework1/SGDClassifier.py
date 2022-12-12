from CustomLR import CustomLR
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pickle


class BuildInLogClassifier(CustomLR):
    def __init__(self, path: str, lr: float = 0.01, itr: float = 5000, lmda: float = 1000, reg: str = 'L2', feature_type: str = 'Bag') -> None:
        self.model_path = 'build-in-classifier.json'
        self.loss = 'log_loss'
        super().__init__(path, lr, itr, lmda, reg, feature_type)

    def load_model(self, model_path: str, model: dict) -> None:
        try:
            super().load_model(model_path, pickle.load(open(model_path, 'rb')))
        except Exception as e:
            super().load_model(model_path, model)

    def classify(self, data: tuple):
        x_train = np.zeros((len(self.uniq_words)))
        for each in data.keys():
            try:
                i = self.uniq_words.index(each)
                x_train[i] = 1
            except Exception as e:
                pass
        return self.model.predict([x_train])

    def post_process(self, data: tuple):
        return super().post_process(data[0])

    def fit(self, x_train: np.array, y_true: np.array):
        self.model = SGDClassifier(
            max_iter=self.itr, alpha=self.lmda, learning_rate=self.lr, loss=self.loss, tol=1e-3)
        self.model.fit(x_train, y_true)
        pickle.dump(self.model_path, open(self.model_path, 'wb'))

    def get_parameters(self, x_train, train_label, val_cls_data, val_label):
        # lamda to match my code
        lamda = [0, 1/(2 * 10), 1 / (2 * 100), 1 / (2 * 500), 1/(2 * 1000)]
        itrs = [100, 500, 1000]
        parameters = {'alpha': lamda, 'max_iter': itrs,  'penalty': ['l2'], 'learning_rate': [
            'optimal', 'invscaling', 'adaptive'], 'loss': ['log_loss', 'ridge', 'hinge']}
        self.model = SGDClassifier(tol=1e-3)
        clf = GridSearchCV(self.model, param_grid=parameters, scoring='accuracy', n_jobs=-1)
        clf.fit(x_train, train_label)
        self.itr = clf.best_params_['max_iter']
        self.lmda = clf.best_params_['alpha']
        self.lr = clf.best_params_['learning_rate']
        self.loss = clf.best_params_['loss']
        print("Best Hyper Parameters chosen",  clf.best_params_)


# if __name__ == "__main__":
#     # save_unique_words()
#     bofw = BuildInLogClassifier('train_data')
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
