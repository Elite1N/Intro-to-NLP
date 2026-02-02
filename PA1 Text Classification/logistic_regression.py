import sys

import pandas as pd
import pythainlp
import nltk
import math
import numpy as np
# nltk.download('punkt_tab') #???
from nltk.tokenize import word_tokenize
class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0) # index_col uses the first column (word) as index

    def compute_probability(self, text_string):
        
        # Tokenize text_string using nltk
        
        text_string = text_string.lower() # Deal with case sensitivity (ทำไปก่อน)
        tokenized_string = word_tokenize(text_string)
        tokenized_string = set(tokenized_string)
        
        labels = self.get_all_possible_labels()
        features = self.get_all_possible_features()
        input_data = tokenized_string.intersection(features) # input data สำหรับใช้ทำ inference (เอาแค่ตัวที่มีใน feature จาก model.csv)
        
        # Note: feature weight อยู่ใน self.model_params ; stored df
        scores = {}
        for label in labels:
            scores[label] = 0
            for feature in input_data: # loop คูณด้วย weight สำหรับแต่ละ feature ใน input_data (for all target labels)
                weight = self.model_params.loc[feature, label]
                scores[label] += weight
        
        probabilities = {}        
        # Uses softmax to calculate probabilities for all labels
        for label in scores.keys():
            probabilities[label] = round(math.exp(scores[label]) / sum(math.exp(scores[i]) for i in scores), 2)
        
        return probabilities
    
    # These 2 get point to attributes of a pandas DataFrame
    def get_all_possible_features(self):
        return list(self.model_params.index)

    def get_all_possible_labels(self):
        return list(self.model_params.columns)

    def classify(self, text_string):
        probabilities = self.compute_probability(text_string)
        return max(probabilities, key=probabilities.get)


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tpython logistic_regression.py <model_file>')
        sys.exit(0)
    model_file_name = sys.argv[1]
    model = TextClassifier(model_file_name)
    
    # Testing the functions
    print (f"There are {len(model.get_all_possible_features())} features in the model:")
    print (model.get_all_possible_features())
    print("")
    
    print (f"There are {len(model.get_all_possible_labels())} labels:")
    print (model.get_all_possible_labels())
    print("")
    
    print ("Predicted prob for each label")
    print (model.compute_probability("I HATE DUST"))
    print("")
    
    print ("Finally, our predicted label ->", model.classify("I HATE DUST"))

