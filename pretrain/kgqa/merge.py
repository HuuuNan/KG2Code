import json

fact_ques=json.load(open('fact_question.json','r',encoding='utf-8'))
print(len(fact_ques))
count_ques=json.load(open('count_question.json','r',encoding='utf-8'))
print(len(count_ques))
judge_ques=json.load(open('judge_question.json','r',encoding='utf-8'))
print(len(judge_ques))
data=[]
data.extend(fact_ques)
data.extend(count_ques)
data.extend(judge_ques)
print(len(data))
json.dump(data,open('all_question.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)