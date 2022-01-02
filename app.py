from flask import Flask,render_template,request,url_for
#EDA Packages
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import pickle
import re
app = Flask(__name__)
class scibot:
    def __init__(self):
        pass
    def main(df,target,measure):
        strout = ""
        df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        dtypedf = pd.DataFrame()
        dtypedf['Feature'] = df.dtypes.index
        dtypedf['dtype'] = [str(x) for x in df.dtypes]
        numdf = df.loc[:,df.dtypes!=np.object]
        catdf = df.loc[:,df.dtypes==np.object]
        cat = 0
        if df[target].dtypes==np.object:
            cat=1
        strn="There are "+str(df.shape[0])+" rows and "+str(df.shape[1])+"columns in the Data. "
        strn=strn + " There are xy Numerical features in dataframe. "
        strn = strn.replace("xy",str(len(numdf.columns)))
        strout = strout +strn
        strn="There are xy Categorical features in dataframe. "
        strn = strn.replace("xy",str(len(catdf.columns)))
        strout = strout +strn
        if(df.isnull().sum().sum()>0):
            strn = "There are "+str(df.isnull().sum().sum())+" Missing values in the data. Which will be Replaced by Mean Mode Imputaion Method."
            strout = strout +strn
        else:
            strn = "There are no Missing values in data"
            strout = strout +strn
        for i in numdf.columns:
            df[i].fillna(df[i].mean(skipna=True),inplace=True)
        for i in catdf.columns:
            df[i].fillna(df[i].mode()[0],inplace=True)
        for i in catdf.columns:
            le = LabelEncoder()
            le.fit(df[i])
            df[i]= pd.Series(le.transform(df[i]))
        results = pd.DataFrame({"Model":[],"Accuracy":[],"AUC-ROC":[],"Precission":[],"Recall":[],"F1":[]})
        from sklearn.tree import DecisionTreeClassifier
        X = df.drop(columns = target)
        y =df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        results.loc[len(results.index)] = ['DecisionTreeClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.ensemble import RandomForestClassifier
        X = df.drop(columns = target)
        y =df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['RandomForestClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['ExtraTreesClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        clf = LGBMClassifier()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['LGBMClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.ensemble import GradientBoostingClassifier
        clf  = GradientBoostingClassifier()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['GradientBoostingClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.ensemble import AdaBoostClassifier
        clf= AdaBoostClassifier(n_estimators=50)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['AdaBoostClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['QuadraticDiscriminantAnalysis', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['GaussianNB', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=len(y.value_counts()))
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['KNeighborsClassifier', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.linear_model import Ridge, Lasso
        clf = Ridge()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        y_pred = np.round(y_pred)
        results.loc[len(results.index)] = ['Ridge', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        clf = Lasso()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        y_pred = np.round(y_pred)
        results.loc[len(results.index)] = ['Lasso', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results.loc[len(results.index)] = ['LinearDiscriminantAnalysis', metrics.accuracy_score(y_test, y_pred), metrics.roc_auc_score(y_test, y_pred),
                                 metrics.precision_score(y_test, y_pred),metrics.recall_score(y_test, y_pred),metrics.f1_score(y_test, y_pred)]
        results.sort_values(by =measure,inplace = True,ascending=False)
        return results
    
 
@app.route("/")
def index():
	return render_template("index.html")

def guess (dataset):
    index = find(results)
    return index

@app.route("/data",methods=['POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        target = request.form['target']
        measure = int(request.form['measure'])
        with open(f) as file:
            csvfile = pd.read_csv(file)
        mes = ['Accuracy', 'AUC-ROC', 'Precission', 'Recall', 'F1']
        a = scibot.main(csvfile,str(target),mes[measure])
        return render_template('data.html',data =a.to_html())
if __name__ == '__main__':
    app.run(debug=True)