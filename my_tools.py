# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 15:56:28 2018

@author: Henning
"""

import pandas as pd
import numpy as np

def feature_stats(data_dict):
    "lists statistics on features"
    
    entry_no = len(data_dict.keys())
    
    features = data_dict['YEAP SOON'].keys()
    
    features.remove('poi')
    features.remove('email_address')
    
    columns = ['available_no','available_pct','max','min','mean','median']
    
    feature_stats = pd.DataFrame(index=features, columns=columns)
    
    for feature in features:
        feature_stats.loc[feature]['available_no'] = 0
        value = []
        for person in data_dict:
            if data_dict[person][feature]!='NaN':
                feature_stats.loc[feature]['available_no'] += 1
                value.append(data_dict[person][feature])
        feature_stats.loc[feature]['available_pct'] = round((feature_stats.loc[feature]['available_no'] / float(entry_no)) * 100, 2)
        feature_stats.loc[feature]['max'] = max(value)
        feature_stats.loc[feature]['min'] = min(value)
        feature_stats.loc[feature]['mean'] = np.mean(value)
        feature_stats.loc[feature]['median'] = np.median(value)
        
    return feature_stats

def new_feature(data_dict):
    "creates new feature containing the ratio between long term incentive and salary and long term incentive and bonus respectively"
    
    my_data_dict = data_dict
    
    for person in my_data_dict.keys():
        ratio_1 = .1
        ratio_2 = .1
        if isinstance(my_data_dict[person]['long_term_incentive'], (int, long)) and \
        isinstance(my_data_dict[person]['salary'], (int, long)) and my_data_dict[person]['salary'] != 0:
            ratio_1 = 100-(my_data_dict[person]['long_term_incentive'] / float(my_data_dict[person]['salary']))
        else: ratio_1 = 'NaN'
            
        if isinstance(my_data_dict[person]['long_term_incentive'], (int, long)) and \
        isinstance(my_data_dict[person]['bonus'], (int, long)) and my_data_dict[person]['bonus'] != 0:
            ratio_2 = 100-(my_data_dict[person]['long_term_incentive'] / float(my_data_dict[person]['bonus']))
        else: ratio_2 = 'NaN'
        
        #print 'ratio_1: ', ratio_1
        #print 'ratio_2: ', ratio_2
        
        my_data_dict[person]['long_term_incentive_vs_salary'] = ratio_1
        my_data_dict[person]['long_term_incentive_vs_bonus'] = ratio_2

    return my_data_dict

def best_features(data_dict, features_list):
    "determine best features using SelectKBest"
    
    import pandas as pd
    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    
    clf = SelectKBest(f_classif, k='all').fit(features, labels)
    
    return pd.DataFrame({'scores' : clf.scores_}, index=features_list[1:]).sort_values(ascending=False, by='scores')

def rescale_data_dict(data_dict, features_list):
    "Rescale relevant features in data dict for use with algorithms relying on distances"
    
    from sklearn.preprocessing import MinMaxScaler
    #from sklearn.preprocessing import StandardScaler
    
    from feature_format import featureFormat
    
    data = featureFormat(data_dict, features_list, remove_all_zeroes=False)
    #print len(data)
    
    clf = MinMaxScaler()
    #clf = StandardScaler()
    
    data_rescaled = clf.fit_transform(data)
    
    ii = 0
    
    for person in data_dict.keys():
        jj = 0
        for feature in features_list:
            if feature in data_dict[person].keys():
                if data_dict[person][feature] != 'NaN':
                    data_dict[person][feature] = data_rescaled[ii][jj]
                jj += 1
        ii += 1
    
    return data_dict


def tune_k(data_dict, features_list, clf):
    '''
    increase number of used features k and determine performance of classifier
    starting with the features having the highest scores as found by SelectKBest
    '''
    import pandas as pd
    from feature_format import featureFormat, targetFeatureSplit
    
    #from sklearn.naive_bayes import GaussianNB
    from sklearn.cross_validation import train_test_split
    from my_tester import test_classifier
    
    scores = []
    
    for k in range(len(features_list))[1:]:

        ### Extract features and labels from dataset for local testing
        data = featureFormat(data_dict, features_list[:k+1], sort_keys = True)
        labels, features = targetFeatureSplit(data)
    
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=None, random_state = 42)
    
        clf = clf.fit(train_features, train_labels)
    
        # validate by Udacity test procedure and return in pandas dataframe
        columns = ['features used','total predictions','accuracy','precision','recall','F1']
        scores.append([k]+test_classifier(clf, data_dict, features_list[:k+1]))
        
    scores = pd.DataFrame(columns=columns, data=scores)
    
    return scores, features

def tune_k_nn(data_dict, features_list, n_features, n_neighbors):
    from sklearn.neighbors import KNeighborsClassifier
    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.cross_validation import train_test_split
    from tester import test_classifier
    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='auto')
    data_dict = rescale_data_dict(data_dict, features_list)
    
    data = featureFormat(data_dict, features_list[:n_features+1], sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=None, random_state = 42)
    
    clf = clf.fit(train_features, train_labels)
    
    test_classifier(clf, data_dict, features_list[:n_features+1])