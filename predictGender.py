#!/usr/bin/env python
# encoding: utf-8
"""
predictGender
"""
import pickle
import csv
import random
from nltk import NaiveBayesClassifier,classify


"""
read names.pickle
"""
class predictGender():
    def loadNamesFromPickle(self):
        f = open('names.pickle', 'rb')
        maleNames, femaleNames = pickle.load(f)
        finalnames = dict()
        for name in maleNames:
            finalnames[name[0]] = 'M'
        for name in femaleNames:
            finalnames[name[0]] = 'F'
        return finalnames
    
    def writeNames(self, names):
        f = open('train.csv', 'w')
        try:
            csv_writer = csv.writer(f)
            for name in names.keys():
                csv_writer.writerow((name, names[name]))
        finally:
            f.close()
    
    def generateFeatures(self):
        f = open('train.csv', 'r')
        try:
            csv_reader = csv.reader(f)
            trainfeatureset = list()
            for name in csv_reader:
                features = {'f1': name[0][-1], 'f2': name[0][-2:], 'f3': (name[0][-1] in 'AEIOU'), 'f4': name[0][1]}
                trainfeatureset.append((features, name[1]))
            random.shuffle(trainfeatureset)
            self.train_set = trainfeatureset
        finally:
            f.close()
            
    def generateTestData(self):
        f = open('test.csv', 'r')
        try:
            csv_reader = csv.reader(f)
            testfeatureset = list()
            for name in csv_reader:
                features = {'f1': name[1][-1], 'f2': name[1][-2:], 'f3': (name[1][-1] in 'AEIOU'), 'f4': name[1][1]}
                testfeatureset.append((features, name[0]))
            self.test_set = testfeatureset
        finally:
            f.close()
    
if __name__ == "__main__":
    predictGender = predictGender()
#to generate training data from names.pickle
#     names = predictGender.loadNamesFromPickle()
#     predictGender.writeNames(names)
    
    
    predictGender.generateFeatures()
    classifier = NaiveBayesClassifier.train(predictGender.train_set)
    #form the test data
    predictGender.generateTestData()
    print classify.accuracy(classifier, predictGender.test_set)
    