from flask import Flask
from flask_restful import Api, Resource 
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from loader import *
import pandas as pd
import os
import urllib.parse
app=Flask(__name__) 
api=Api(app)
CORS(app)

#đưa id vào database để trả về title
News_data=pd.read_csv(os.path.join("NewsAPI", "NewsVN.csv"), encoding="utf8")
News_data.set_index("id", inplace=True)

class Vectorspace(Resource):
    def get(self,query):
        path_PL="NewsAPI\Posting_List_News.pkl" #path to PL here
        path_vocab=os.path.join("NewsAPI", "vocab.csv") # path to vocab here
        PL,vocab=importVectorModel(path_PL,path_vocab)
        idd,results=query_results_vector(query,PL,vocab)
        title=[]
        for i in range(len(idd)):
            title.append([News_data.loc[idd[i]]["title"],News_data.loc[idd[i]]["link"], News_data.loc[idd[i]]["description"]])
        return title

api.add_resource(Vectorspace,"/VectorSpaceModel/<string:query>")

if __name__=="__main__":
    app.run(debug=True)
