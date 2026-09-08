import json
import re
import os
from tqdm import tqdm

input_prompt='''Please answer the question based on the subgraph retrieved from the knowledge graph. First, provide your Chain-of-Thought (CoT) reasoning process. At the end, list all answers in a single line, starting with "Answer: " and separating each answer by "|" as follows: Answer: First answer|Second answer|Third answer ...
Subgraph: {knowledge}
Question: {ques}
'''

def extract_triplets(code_str):
    """
    Extract all (h, r, t) triplets from the provided code string.
    
    Parameters:
    - code_str: The string of code from which to extract the triplets.
    
    Returns:
    - A list of tuples representing the triplets (h, r, t).
    """
    
    # Regular expression to match the add_edge method calls
    edge_pattern = r'graph\.add_edge\("([^"]+)", "([^"]+)", relation="([^"]+)"\)'
    
    # Find all matches in the code string
    matches = re.findall(edge_pattern, code_str)
    
    # Create the list of triplets (h, r, t)
    triplets = [[h, r, t] for h, t, r in matches]
    
    return triplets

def extract_question(text):
    """
    Extract the question from the provided string.
    Assumes the question comes after the 'question = ' part of the string.
    """
    # Use a regular expression to extract everything after 'question = '
    match = re.search(r'question\s*=\s*\'(.*)\'', text)
    
    # If a match is found, return the question text
    if match:
        return match.group(1)
    else:
        return None  # Return None if no question is found

def extract_comments(code):
    lines = code.splitlines()
    comments = []
    current_comment = ""

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            if current_comment:
                current_comment += " " + line[1:].strip()
            else:
                current_comment = line[1:].strip()
        else:
            if current_comment:
                comments.append(current_comment)
                current_comment = ""

    if current_comment:  # Append the last collected comment if any
        comments.append(current_comment)

    return comments

def replace_list_with_pipe_separated(input_str):
    """
    Replace list-like structures in the string with pipe-separated values.
    
    Args:
    input_str (str): A string that may contain list-like structures (e.g., '["apple", "banana"]').
    
    Returns:
    str: The input string with list-like structures replaced by pipe-separated values.
    """
    # Regular expression pattern to match list-like structures
    pattern = r'\[(.*?)\]'
    
    # Function to replace the list structure with a pipe-separated string
    def convert_match(match):
        # Extract the content inside the brackets
        list_content = match.group(1)
        
        # Split the content by comma and strip the extra spaces, then join by '|'
        items = [item.strip().strip('"') for item in list_content.split(',')]
        return '|'.join(items)
    
    # Replace all occurrences of list-like structures with the pipe-separated format
    result = re.sub(pattern, convert_match, input_str)
    
    return result

data=json.load(open('code_corpus/train/train.json','r',encoding='utf-8'))
text_corpus=[]
for sample in tqdm(data):
    t_list=extract_triplets(sample["instruction"])
    t_string=''
    for t in t_list:
        t_string+=f"({t[0]}, {t[1]}, {t[2]}) "
    t_string=t_string.strip()
    ques=extract_question(sample["instruction"])
    c_list=extract_comments(sample["output"])
    c_list=[replace_list_with_pipe_separated(c) for c in c_list]
    s_id=0
    for index,c in enumerate(c_list):
        if c.startswith('Step'):
            s_id=index
            break
    answer=c_list[-1].split(':',1)[-1].strip()
    comment_cot='\n'.join(c_list[s_id:]).strip()
    text_s=dict()
    text_s["instruction"]=input_prompt.format(knowledge=t_string,ques=ques)
    text_s["input"]=""
    text_s["output"]=comment_cot+'\n'+'Answer: '+answer
    text_corpus.append(text_s)

os.makedirs('text_corpus/train',exist_ok=True)
json.dump(text_corpus,open('text_corpus/train/train.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)

data=json.load(open('code_corpus/dev.json','r',encoding='utf-8'))
text_corpus=[]
for sample in tqdm(data):
    t_list=extract_triplets(sample["instruction"])
    t_string=''
    for t in t_list:
        t_string+=f"({t[0]}, {t[1]}, {t[2]}) "
    t_string=t_string.strip()
    ques=extract_question(sample["instruction"])
    c_list=extract_comments(sample["output"])
    c_list=[replace_list_with_pipe_separated(c) for c in c_list]
    s_id=0
    for index,c in enumerate(c_list):
        if c.startswith('Step'):
            s_id=index
            break
    answer=c_list[-1].split(':',1)[-1].strip()
    comment_cot='\n'.join(c_list[s_id:]).strip()
    text_s=dict()
    text_s["instruction"]=input_prompt.format(knowledge=t_string,ques=ques)
    text_s["input"]=""
    text_s["output"]=comment_cot+'\n'+'Answer: '+answer
    text_corpus.append(text_s)

json.dump(text_corpus,open('text_corpus/dev.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
    