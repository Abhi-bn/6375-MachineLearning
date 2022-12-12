import os
from Bernoulli import Bernoulli
from BagOfWords import BagOfWords
from CustomLR import CustomLR
from SGDClassifier import BuildInLogClassifier
import os
all_train_dataset = {
    'enron1': {
        'ham': 'enron1/train/ham',
        'spam': 'enron1/train/spam'
    },
    'enron4': {
        'ham': 'enron4/train/ham',
        'spam': 'enron4/train/spam'
    },
    'hw1': {
        'ham': 'hw1/train/ham',
        'spam': 'hw1/train/spam'
    }
}

all_test_dataset = {
    'enron1': {
        'ham': 'enron1/test/ham',
        'spam': 'enron1/test/spam'
    },
    'enron4': {
        'ham': 'enron4/test/ham',
        'spam': 'enron4/test/spam'
    },
    'hw1': {
        'ham': 'hw1/test/ham',
        'spam': 'hw1/test/spam'
    }
}

if __name__ == "__main__":
    try:
        os.mkdir('train_data')
    except:
        pass
    # Run on all dataset
    for k, v in all_train_dataset.items():
        print("=========Bernoulli", k, "===========")
        ber = Bernoulli('train_data')
        ber.train_data(v)
        ber.test_data(all_test_dataset[k])
        print("====Done[Bernoulli", k, "]==========")

    for k, v in all_train_dataset.items():
        print("========BagOfWords", k, "===========")
        bag = BagOfWords('train_data')
        bag.train_data(v)
        bag.test_data(all_test_dataset[k])
    #     print("===Done[BagOfWords", k, "]==========")

    for k, v in all_train_dataset.items():
        print("=Bag Logistic Regression:", k, "===========")
        clr = CustomLR('train_data', feature_type='Bag')
        clr.cross_validation(v)
        clr.train_data(v)
        clr.test_data(all_test_dataset[k])
        print("===Done[BagOfWords", k, "]==========")

    for k, v in all_train_dataset.items():
        print("=Ber Logistic Regression:", k, "===========")
        clr = CustomLR('train_data', feature_type='Ber')
        # clr.cross_validation(v)
        clr.train_data(v)
        clr.test_data(all_test_dataset[k])
        print("===Done[BagOfWords", k, "]==========")

    for k, v in all_train_dataset.items():
        print("=sklearn Logistic Regression:", k, "===========")
        clr = BuildInLogClassifier('train_data', feature_type='Ber')
        clr.cross_validation(v, True)
        clr.train_data(v)
        clr.test_data(all_test_dataset[k])
        print("===Done[BagOfWords", k, "]==========")

    for k, v in all_train_dataset.items():
        print("=sklearn Logistic Regression:", k, "===========")
        clr = BuildInLogClassifier('train_data', feature_type='Bag')
        clr.cross_validation(v, True)
        clr.train_data(v)
        clr.test_data(all_test_dataset[k])
        print("===Done[BagOfWords", k, "]==========")
