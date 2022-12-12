from cProfile import label
from BagOfWords import BagOfWords
import numpy as np
from glob import glob
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
import matplotlib.pyplot as plt
import csv
# suppress warnings
warnings.filterwarnings('ignore')


class CustomLR(BagOfWords):
    def __init__(self, path: str, lr: float = 0.01, itr: float = 5000, lmda: float = 1000, reg: str = 'L2', feature_type: str = 'Bag') -> None:
        assert reg == 'L2' or reg == 'L1', "only L2 and L1 supported"
        assert feature_type == 'Bag' or feature_type == 'Ber', "Only Bag of words and Bernoulli features Supported"

        self.model_path = 'LogisticRegression.json'
        self.model = {}
        self.lr = 0.1
        self.itr = 500
        self.lmda = 0
        self.reg = reg
        self.feature_type = feature_type
        self.all_results = {}
        try:
            with open(path + "/" + self.model_path, 'r') as f:
                super().load_model(path + "/" + self.model_path, json.load(f))
        except Exception as e:
            super().load_model(path + "/" + self.model_path, self.model)

    def pre_process(self, data: tuple):
        return BagOfWords.get_words(data, self.feature_type == "Bag")

    def save_model(self):
        self.model = {}
        self.model['w0'] = self.bias
        self.model['lr'] = self.lr
        self.model['w'] = list(self.weights)
        self.model['dictionary'] = self.uniq_words
        self.model['itr'] = self.itr
        self.model['cost'] = self.costs
        with open(self.model_path, 'w') as f:
            json.dump(self.model, f)

    def cross_validation(self, data: dict, full: bool = False):
        train_label = []
        val_label = []
        train_cls_data = []
        val_cls_data = []

        store = train_cls_data
        label = train_label

        self.uniq_words = set()

        for index, key in enumerate(data):
            this_class_data = glob(data[key] + "/*.*")

            for ind, path in enumerate(this_class_data):
                if not full:
                    val = len(this_class_data) * 0.7
                    if ind > val:
                        store = val_cls_data
                        label = val_label
                    else:
                        store = train_cls_data
                        label = train_label

                with open(path, 'r', errors='ignore') as f:
                    this_class_words = {}
                    for new_wd, c in self.pre_process(f.read()).items():
                        this_class_words.setdefault(new_wd, 0)
                        this_class_words[new_wd] += c
                        self.uniq_words.add(new_wd)
                    store.append(this_class_words)
                    label.append(index)

        self.uniq_words = sorted(self.uniq_words)
        x_train = np.zeros((len(train_cls_data), len(self.uniq_words)))
        for index, each in enumerate(train_cls_data):
            for k in each.keys():
                i = self.uniq_words.index(k)
                x_train[index][i] = 1

        self.get_parameters(x_train, train_label, val_cls_data, val_label)

    def get_parameters(self, x_train, train_label, val_cls_data, val_label):
        lamda = [0, 10, 100, 500, 1000]
        lrs = [0.1, 0.01, 0.001]
        itrs = [100, 500, 1000]
        cost_func = {}

        for itr in itrs:
            for lr in lrs:
                for lmd in lamda:
                    title = 'i:'+str(itr)+';lr:'+str(lr)+';lmd:'+str(lmd)
                    self.lmda = lmd
                    self.lr = lr
                    self.itr = itr
                    self.fit(x_train, np.array(train_label))
                    cost_func[title] = self.costs
                    self.test_val_data(val_cls_data, val_label)

                    with open("hw-files/all_results.csv", 'w') as f:
                        csv_reader = csv.DictWriter(f, fieldnames=[
                                                    'itr', 'lr', 'lambda', 'train_accuracy', 'final_loss', 'recall_score', 'precision_score', 'accuracy_score', 'f1_score'], delimiter=',')
                        for values in self.all_results.values():
                            csv_reader.writerow(values)
        m = max(self.all_results, key=lambda x: self.all_results[x]['accuracy_score'])
        print("Best Hyper Parameters chosen", self.all_results[m])
        self.lmda = int(self.all_results[m]['lambda'])
        self.lr = float(self.all_results[m]['lr'])
        self.itr = int(self.all_results[m]['itr'])

    def train_data(self, data: dict):
        self.uniq_words = set()
        label = []
        each_cls_data = []
        for index, key in enumerate(data):
            this_class_data = glob(data[key] + "/*.*")
            c_words = 0

            for path in this_class_data:
                with open(path, 'r', errors='ignore') as f:
                    this_class_words = {}
                    for new_wd, c in self.pre_process(f.read()).items():
                        this_class_words.setdefault(new_wd, 0)
                        this_class_words[new_wd] += c
                        self.uniq_words.add(new_wd)
                        c_words += c
                    each_cls_data.append(this_class_words)
                    label.append(index)

        self.uniq_words = sorted(self.uniq_words)
        x_train = np.zeros((len(each_cls_data), len(self.uniq_words)))
        for index, each in enumerate(each_cls_data):
            for k in each.keys():
                i = self.uniq_words.index(k)
                x_train[index][i] = 1

        self.fit(x_train, np.array(label))

    def loss_function(self, y_true, y_pred):
        _1_loss = y_true * np.log(y_pred)
        _0_loss = (1-y_true) * np.log(1 - y_pred)
        out = -np.sum(_1_loss + _0_loss)
        if self.lmda != 0 and self.reg == 'L2':
            out -= (1 / (2 * self.lmda)) * np.dot(self.weights, self.weights)
        return out

    def compute_grad(self, y_true, y_pred, x):
        g_weight = np.matmul(x.transpose(), y_pred - y_true)
        return g_weight

    def classify(self, data: tuple):
        x_train = np.zeros((len(self.uniq_words)))
        for each in data.keys():
            try:
                i = self.uniq_words.index(each)
                x_train[i] = 1
            except Exception as e:
                pass

        mul = self.weights[0] + np.matmul(x_train, self.weights[1:])
        prob = 1 / (1 + np.exp(-mul))
        return prob

    def post_process(self, data: tuple):
        return 1 if data > 0.5 else 0

    def save_all_results(self, res: dict):
        it = str(self.itr)
        l = str(self.lmda)
        lr = str(self.lr)
        self.all_results.setdefault(it+l+lr, {})
        self.all_results[it+l+lr]['lr'] = lr
        self.all_results[it+l+lr]['itr'] = it
        self.all_results[it+l+lr]['lambda'] = l
        for k, v in res.items():
            self.all_results[it+l+lr][k] = v

    def fit(self, x_train: np.array, y_true: np.array):
        self.weights = np.zeros(x_train.shape[1] + 1)
        x_train = np.c_[np.ones([np.shape(x_train)[0], 1]), x_train]
        self.bias = 0
        self.costs = []
        for i in range(self.itr):
            mul = np.dot(x_train, self.weights)
            y_pred = 1 / (1 + np.exp(-mul))

            loss = self.loss_function(y_true, y_pred)

            dw = np.matmul(x_train.transpose(), y_true - y_pred)

            if self.lmda != 0 and self.reg == 'L2':
                dw -= (1 / (2 * self.lmda)) * np.dot(self.weights, self.weights)

            if self.lmda != 0 and self.reg == 'L1':
                dw -= (1 / (2 * self.lmda)) * self.weights

            self.weights += (1 / x_train.shape[0]) * (self.lr * dw)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            if i % (self.itr // 10) == 0:
                print("itr: ", i, " accuracy: ", accuracy_score(y_true, y_pred), " loss: ", loss)
            self.costs.append(loss)

            if np.isnan(loss):
                break

        # self.save_model()
        self.save_all_results({
            "train_accuracy": accuracy_score(y_true, y_pred),
            "final_loss": loss
        })

    def plotCost(self):
        """Plot value of log-liklihood cost function for each epoch

        Returns
        --------
        matplotlib figure

        """
        plt.figure()
        plt.plot(np.arange(1, self.itr + 1), self.costs, marker='.')
        plt.xlabel('Iterations')
        plt.ylabel('Log-Liklihood J(w)')
        plt.show()

    def test_val_data(self, x_test: np.array, y_test: np.array):
        Y_actual = []
        Y_predicted = []

        for i, p in enumerate(x_test):
            c = self.post_process(self.classify(p))
            Y_actual.append(y_test[i])
            Y_predicted.append(c)
            # print(y_test[i], c)
        # confusion_matrix = metrics.confusion_matrix(Y_actual, Y_predicted)
        info = {
            "recall_score": metrics.recall_score(Y_actual, Y_predicted),
            "precision_score": metrics.precision_score(Y_actual, Y_predicted),
            "accuracy_score": metrics.accuracy_score(Y_actual, Y_predicted),
            "f1_score": metrics.f1_score(Y_actual, Y_predicted)
        }

        self.save_all_results(info)

    def test_data(self, data):
        Y_actual = []
        Y_predicted = []
        for key, value in data.items():
            this_class_data = glob(data[key] + "/*.*")
            for p in this_class_data:
                with open(p, 'r', errors='ignore') as f:
                    c = self.process(f.read())
                    Y_actual.append(0 if key == 'ham' else 1)
                    Y_predicted.append(c)
        confusion_matrix = metrics.confusion_matrix(Y_actual, Y_predicted)
        print("recall_score,", metrics.recall_score(Y_actual, Y_predicted))
        print("precision_score,", metrics.precision_score(Y_actual, Y_predicted))
        print("accuracy_score,", metrics.accuracy_score(Y_actual, Y_predicted))
        print("f1_score,", metrics.f1_score(Y_actual, Y_predicted))
        # cm_display = metrics.ConfusionMatrixDisplay(
        #     confusion_matrix=confusion_matrix, display_labels=['ham', 'spam'])
        # cm_display.plot()
        # plt.show()


if __name__ == "__main__":
    # save_unique_words()
    bofw = CustomLR('train_data')
    # bofw.cross_validation(
    #     {'ham': 'enron1/train/ham', 'spam': 'enron1/train/spam'})
    bofw.train_data(
        {'ham': 'enron1/train/ham', 'spam': 'enron1/train/spam'})

    bofw.test_data(
        {'ham': 'enron1/test/ham', 'spam': 'enron1/test/spam'})
