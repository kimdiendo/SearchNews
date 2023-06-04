import pickle
import numpy as np
import os
import pandas as pd
from operator import itemgetter
import Preprocessor
import math

def importVectorModel(path_PL,path_vocab):#import posting list và vocab

    with open(path_PL, 'rb') as f:
        data = pickle.load(f)
    
    vocab=pd.read_csv(path_vocab,encoding="utf-8")
    vocab.set_index("Key", inplace=True)
    return data, vocab
def weightingQ(query_term):#đặt weight cho query term
    sum_w = 0
    for term in query_term:
        sum_w += math.pow(term[1] , 2)
    sum_w = math.sqrt(sum_w)
    for term in query_term:
        term[1]=term[1]/sum_w
    return query_term
  
def find_and_add(arr,doc_id,value):
  for i in range(len(arr)):
    if arr[i][0]== doc_id:
         arr[i][1]+=value
  return arr
def search(weightedQ,PL):#truy suất
  q=weightedQ
  results=[]
  for key in q:
    for j in PL[key[0]]:
      results.append([j[0],j[3]*key[1]])
  doc_rank=[]#lưu trữ (id duy nhất , weight) của từng truy vấn
  check=[]
  for result in results:
    doc_id = result[0]
    if doc_id not in check:
      check.append(doc_id)
      doc_rank.append(result)
    else:
      doc_rank=find_and_add(doc_rank,doc_id,result[1])
  arr=sorted(doc_rank, key=itemgetter(1))#sắp xếp doc_rank theo thứ tự từ bé đến lớn
  return(arr[-10:])
def query_results_vector(query,PL,vocab):
  query_term=Preprocessor.text_preprocess(query)
  check=query_term.copy()
  unused_term=[]
  for i in range(len(query_term)):#duyệt qua từng queries
    term=query_term[i] #xác định term
    query_term[i]=[term] 
    try: 
      query_term[i].append((check.count(term)/len(check))*(vocab.loc[term]["IDF"]))
    except KeyError:
      print(term,"is not a key",i)
      unused_term.append(term)
  weightedQ=weightingQ(query_term)
  weightedQ=search(weightedQ,PL)
  idd=[]

  for i in range(len(weightedQ)-1,-1,-1):
    idd.append(weightedQ[i][0])
  return idd,weightedQ
