from flask import Flask
from flask_restful import Api, Resource 
from flask_sqlalchemy import SQLAlchemy
from loader import *
import pandas as pd
import os
import urllib.parse
app=Flask(__name__) 
api=Api(app)

#đưa id vào database để trả về title
News_data=pd.read_csv(os.path.join("K:\SearchNews\Data Acquisition","NewsVN.csv"),encoding="utf-8")
News_data.set_index("id", inplace=True)




class Vectorspace(Resource):
    def get(self,query):
        path_PL="D:\SearchNews\Model\Posting_List_News.pkl" #path to PL here
        path_vocab="D:\SearchNews\Model\ vocab.csv" # path to vocab here
        PL,vocab=importVectorModel(path_PL,path_vocab)
        idd,results=query_results_vector(query,PL,vocab)
        title=[]
        for i in range(len(idd)):
            title.append([News_data.loc[idd[i]]["title"],News_data.loc[idd[i]]["link"]])
        
        return title

api.add_resource(Vectorspace,"/VectorSpaceModel/<string:query>")


if __name__=="__main__":
    app.run(debug=True)
