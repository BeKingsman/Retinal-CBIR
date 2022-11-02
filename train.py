from distutils.log import error
from main import *
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from scipy import optimize
from scipy.optimize import minimize

images_path="/Users/harshagarwal/Desktop/ML_SOP/code/images"
os.chdir(images_path)

categories = {}

db = get_db_features()

data = {}



NO_OF_ITERATIONS = 100


for img in os.listdir():
        cat = "".join(img.split("-")[:-1])
        if os.path.join(os.getcwd(),img) not in data.keys():
                data[os.path.join(os.getcwd(),img)]={}
        data[os.path.join(os.getcwd(),img)]["category"]=cat


for x in db:
        img=x["source"]
        data[img]["features"]=x

def create_dataset():
        global weights_array
        fin = []

        for i in range(NO_OF_ITERATIONS):
                a = np.arange(0, 100)
                weights_array = np.random.choice(a, len(weights_array), replace=False)/100

                errors = []
                for img1 in random.sample(data.keys(), 4):
                        try:
                                res=find_distance(data[img1]["features"],db,weights_array=weights_array)
                                for val in res:
                                        if(data[img1]["category"]==data[val[1]]["category"]):
                                                errors.append(val[0])
                                        else:
                                                errors.append(2-val[0])
                                
                        except Exception as e:
                                print(str(e))
                
                temp=weights_array
                temp = np.append(temp,sum(errors)/len(errors))
                fin.append(temp)
                
                print(temp)
                print("Finished Iteration "+str(i+1),end="\n\n")


                
        fin = pd.DataFrame(fin)
        fin.to_csv("/Users/harshagarwal/Desktop/ML_SOP/code/regression_data.csv", encoding='utf-8', index=False)
        print("Completed\n\n")
        



def train():
        df = pd.read_csv("/Users/harshagarwal/Desktop/ML_SOP/code/regression_data.csv",names=["0","1","2","3","4","5","6","7","Loss"])
        X = df[["0","1","2","3","4","5","6","7"]]



def predict_category(res):
        freq = {}
        mx=0
        mx_freq=""
        for val in res:
                cat = data[val[1]]["category"]
                if cat in freq.keys():
                        freq[cat]+=1
                else:
                        freq[cat]=1

                if freq[cat]>=mx:
                        mx = freq[cat]
                        mx_freq = cat
        confidence = mx/len(res)
        return mx_freq,confidence


def find_min_util(arr):
        scores = []

        for img1 in random.sample(data.keys(), 10):
                try:
                        res=find_distance(data[img1]["features"],db,weights_array=arr)
                        cat,confidence = predict_category(res[:30])
                        if data[img1]["category"]== cat:
                                scores.append(1-confidence)
                        else:
                                scores.append(confidence)
                        
                except Exception as e:
                        print(str(e))
        
        # print("find_min_util executed\n weights - ",end="")
        # print(arr)
 


def find_min():
        grid = np.random.choice(np.arange(0, 100), len(weights_array), replace=False)/100
        # xmin_global = optimize.brute(find_min_util, grid)
        res = minimize(find_min_util, grid, method='BFGS',options={'disp': True})
        print("Global minima found "+str(res))


# create_dataset()
# train()
find_min()