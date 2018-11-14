# DMOZ URL Classification based on All-gram method with Python
There is DMOZ URL Classifier Project based on Research Paper (http://www.academia.edu/610148/A_Comprehensive_Study_of_Features_and_Algorithms_for_URL-based_Topic_Classification).

DMOZ dataset contains  training data (category and URI) and testing data (URI only). The URI is represented in all-gram (4-5-6-7-8-gram) combined.

The number of test and train dataset is based on the Research Paper method, which is for testing 1K for each topic, for training the same number of positive (in the category), and same number of negative from all the other categories (not in topic). For example 1000 are in news category we will have to collect 1000/(number of categories) from each category.

The resulted should be a table matching the table in the Research Paper page 10. So for ODP dataset each category has a P, R, and F score with the total average.

To use DMOZ URL Classifier firstly you need to download DMOZ Dataset 'content.rdf.u8' from website 'http://dmoztools.net'
and put this file into project directory 'data/'.
 