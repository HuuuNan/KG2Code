import re
import ast
import os
import json
import random
random.seed(42)

# code corpus
# main corpus
main=json.load(open('../code_corpus/all.json','r',encoding='utf-8'))

# cvt corpus
cvt=json.load(open('../cvt/code_corpus/all.json','r',encoding='utf-8'))

# two intention corpus
two_intention=json.load(open('../two_intention/code_corpus/all.json','r',encoding='utf-8'))

# two intention cvt corpus
two_intention_cvt=json.load(open('../two_intention/cvt/code_corpus/all.json','r',encoding='utf-8'))

# compare corpus
compare=json.load(open('../compare/code_corpus/all.json','r',encoding='utf-8'))

# letter corpus
letter=json.load(open('../letter/code_corpus/all.json','r',encoding='utf-8'))

# time corpus
time=json.load(open('../time/code_corpus/all.json','r',encoding='utf-8'))

# merge the corpus
code_corpus=main+cvt+two_intention+two_intention_cvt+compare+letter+time

def extract_string_list(text):
    """
    Check whether a string contains a string-list-like substring.
    If found, return the list; otherwise, return None.
    """

    # Regular expression to match content inside square brackets
    pattern = r'\[[^\[\]]*\]'

    # Find all substrings that look like lists
    candidates = re.findall(pattern, text)

    for candidate in candidates:
        try:
            # Safely evaluate the string to a Python object
            value = ast.literal_eval(candidate)

            # Check if the evaluated object is a list of strings
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return value

        except (ValueError, SyntaxError):
            # Ignore invalid list formats
            continue

    # Return None if no valid string list is found
    return None

def check_cvt_list(items):
    """
    Check whether a list is:
    1) entirely composed of 'CVT' followed by digits (e.g. CVT1, CVT11), or
    2) entirely NOT composed of 'CVT' followed by digits.

    Mixed cases will return False.

    :param items: list of strings
    :return: True if all match or all do not match, False otherwise
    """

    # Regular expression for 'CVT' followed by one or more digits
    pattern = re.compile(r'^CVT\d+$')

    # Check each item against the pattern
    matches = [bool(pattern.match(item)) for item in items]

    # All True means all match; all False means none match
    return all(matches) or not any(matches)

num=0
# filter extra Question:
code_corpus_filter=[]
for sample in code_corpus:
    # judge whether CVT nodes and entity appear together
    FLAG=False
    for line in sample["output"].split('\n'):
        l_list=extract_string_list(line)
        if l_list is None:
            continue
        if not check_cvt_list(l_list):
            FLAG=True
            break
    if FLAG:
        num+=1
        continue
    sample["instruction"]=sample["instruction"].replace("Question: ","").replace('nx.DiGraph()','nx.MultiDiGraph()')
    code_corpus_filter.append(sample)
random.shuffle(code_corpus_filter)
os.makedirs('code_corpus/train',exist_ok=True)
json.dump(code_corpus_filter,open('code_corpus/all.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
TRAINING_NUM=int(len(code_corpus_filter)*0.9)
json.dump(code_corpus_filter[:TRAINING_NUM],open('code_corpus/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
json.dump(code_corpus_filter[TRAINING_NUM:],open('code_corpus/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

print(num)