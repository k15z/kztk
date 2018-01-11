"""
Analyze comments for NSFW features. Hate speech, etc.
"""
import os
import sys
import pickle
from .model import *

sys.modules['toxic'] = sys.modules["kztk.toxic"]

__dir__ = os.path.dirname(__file__)
path_to_model = os.path.join(__dir__, "model.pkl")
with open(path_to_model, "rb") as fin:
    model = pickle.load(fin)

def classify(text):
    """
    Returns a dictionary indicating the probability of the text being one of the
    following: toxic, severe, insult, obscene, threat, hate

    Examples:

        toxic.classify("flowers and rainbows")
        {
            'threat': 0.0025174026377499104, 
            'insult': 0.01330766174942255, 
            'obscene': 0.011208197101950645, 
            'hate': 0.004297261592000723, 
            'toxic': 0.05422980338335037, 
            'severe': 0.0011627066414803267
        }

        toxic.classify("your post is stupid ")
        {
            'threat': 0.027044642716646194, 
            'insult': 0.2758590579032898, 
            'obscene': 0.32889148592948914, 
            'hate': 0.04638497531414032, 
            'toxic': 0.698431670665741, 
            'severe': 0.0460701622068882
        }
    """
    return model.predict_one([char_vector(s) for s in sent_tokenize(text)])
