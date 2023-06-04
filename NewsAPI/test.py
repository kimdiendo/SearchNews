import requests
BASE="http://127.0.0.1:5000/"
query=input("Tìm kiếm: ")
respond=requests.get(BASE+"VectorSpaceModel/"+query)
print(respond.json()) 