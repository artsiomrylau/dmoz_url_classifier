import re
import gc
import numpy as np
import collections

from dmoz.rdf import RDF
from dmoz.dataset import Dataset
from dmoz.topics import topics
from nltk.util import ngrams
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import precision, recall, f_measure
from nltk.classify.util import accuracy


def get_content():
    """
    Get 'content.csv' file from 'content.rdf.u8'
    :return:
    """
    with RDF('data/content.rdf.u8') as rdf:
        rdf.getPages()
        rdf.showTopics()
        rdf.writeCSV()


def get_categories():
    """
    Get categories csv files from 'content.csv'
    :return:
    """
    ds = Dataset('data/content.csv')
    ds.writeFiles()


def uri_ngram_features(uri, min=4, max=8):
    """
    Get URI 4-5-6-7-8grams Features
    :param uri:
    :param min:
    :param max:
    :return:
    """
    features = {}
    uri_letters = [x for x in re.sub("[^a-zA-Z]", "", uri.split('://')[1]).lower().strip('www')]
    for n in range(min, max + 1):
        try:
            features.update(dict([(gram, True) for gram in ngrams(uri_letters, n)]))
        except StopIteration:
            break
        except DeprecationWarning:
            break
        finally:
            break
    return features


def get_data_set(file_name, max_block_size=1000):
    """
    Get Data Set
    :param file_name:
    :param topic:
    :param max_block_size:
    :return:
    """
    data_set = np.array([])
    with open(file_name, 'r') as file:
        lines = file.readlines()
        block_numbers = np.int32(len(lines) / max_block_size)
        for block in np.arange(0, block_numbers):
            data_list = []
            for line in lines[block * max_block_size: (block + 1) * max_block_size]:
                splitted_line = line.split(',')
                attribute = splitted_line[2].replace("\n", "")
                data_list.append([uri_ngram_features(splitted_line[1]), attribute])
            data_set = np.append(data_set, data_list)
        file.close()
    set_lenght = np.copy(data_set.shape)
    data_set = np.reshape(data_set, (np.int(np.divide(set_lenght, 2)), 2))
    return data_set


def get_train_test_sets(file_path):
    """
    Get Train and Test Data Sets
    :param file_path:
    :param topic:
    :return:
    """
    train_set = get_data_set('{}_{}.csv'.format(file_path, 'train'))
    print('Train data is prepared')
    test_set = get_data_set('{}_{}.csv'.format(file_path, 'test'))
    print('Test data is prepared')
    return train_set, test_set

def main():
    results = {
        'Topic': [],
        'Precision': [],
        'Recall': [],
        'F-measure': []
    }
    print('\nPreparing data...')
    (train_set, test_set) = get_train_test_sets('data/content')
    print('\nNB classifier training...')
    classifier = NaiveBayesClassifier.train(train_set)
    print('NB classifier is trained with {}% accuracy'.format(round(accuracy(classifier, test_set) * 100, 1)))

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    for topic in topics:
        results['Topic'].append(topic)
        results['Precision'].append(round(precision(refsets[topic], testsets[topic]) * 100, 1))
        results['Recall'].append(round(recall(refsets[topic], testsets[topic]) * 100, 1))
        results['F-measure'].append(round(f_measure(refsets[topic], testsets[topic]) * 100, 1))

    del classifier, train_set, test_set, refsets, testsets
    gc.collect()

if __name__ == '__main__':
    get_content()
    get_categories()
    main()
