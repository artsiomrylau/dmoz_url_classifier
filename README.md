# DMOZ URL Classifier based on All-gram method
There is DMOZ URL Classifier Project based on Research Paper (http://www.academia.edu/610148/A_Comprehensive_Study_of_Features_and_Algorithms_for_URL-based_Topic_Classification).

DMOZ dataset contains  training data (category and URI) and testing data (URI only). The URI is represented in all-gram (4-5-6-7-8-gram) combined.

The number of test and train dataset is based on the Research Paper method, which is for testing 1K for each topic, for training the same number of positive (in the category), and same number of negative from all the other categories (not in topic). For example 1000 are in news category we will have to collect 1000/(number of categories) from each category.

The result is table with F-measure, Precision and Recall metrics for every DMOZ category.

The 'dmoz' project package is based on 'https://github.com/gr33ndata/dmoz-urlclassifier' repository.

Before using Project files you need to download DMOZ Dataset 'content.rdf.u8' from website 'http://dmoztools.net'
and put this file into 'data/' Project directory.
 