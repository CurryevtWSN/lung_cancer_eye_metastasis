import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score,classification_report,confusion_matrix,precision_recall_curve
#应用标题
st.set_page_config(page_title='Prediction model for Ocular metastasis of lung cancer')
st.title('Machine Learning for Eye Metastasis of Primary Lung Cancer: development and Verification of Predictive Model')
st.sidebar.markdown('## Variables')
Histopathological_type = st.sidebar.selectbox('Histopathological_type',('Squamous cell carcinoma','Adenocarcinoma','Large cell carcinoma',
                                                                        'Small cell lung cancer','Other non-small cell lung cancer','Unkown'),index=1)
AFP = st.sidebar.slider("AFP", 0.00, 20.00, value=7.00, step=0.01)
CEA = st.sidebar.slider("CEA", 0.00, 1000.00, value=400.00, step=0.01)
CA_125 = st.sidebar.slider("CA_125", 0.00, 2500.00, value=800.00, step=0.01)
CA_199 = st.sidebar.slider("CA_199", 0.00, 2000.00, value=500.00, step=0.01)
CA_153 = st.sidebar.slider("CA_153", 0.00, 500.00, value=200.00, step=0.01)
CYFRA21_1 = st.sidebar.slider("CYFRA21_1", 0.00, 15.00, value=5.00, step=0.01)
TPSA = st.sidebar.slider("TPSA", 0, 2000, value=215, step=1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Squamous cell carcinoma':1,'Adenocarcinoma':2,'Large cell carcinoma':3,'Small cell lung cancer':4,'Other non-small cell lung cancer':5,'Unkown':6}
Histopathological_type =map[Histopathological_type]
# 数据读取，特征标注
hp_train = pd.read_csv('lung_cancer_githubdata.csv')
hp_train['M'] = hp_train['M'].apply(lambda x : +1 if x==1 else 0)
features =["Histopathological_type","AFP","CEA","CA_125","CA_199","CA_153",'CYFRA21_1','TPSA']
target = 'M'
random_state_new = 50
data = hp_train[features]
X_data = data
X_ros = np.array(X_data)
y_ros = np.array(hp_train[target])
oversample = SMOTE(random_state = random_state_new)
X_ros, y_ros = oversample.fit_resample(X_ros, y_ros)
XGB_model = XGBClassifier(n_estimators=360, max_depth=2, learning_rate=0.1,random_state = random_state_new)
XGB_model.fit(X_ros, y_ros)
sp = 0.5
#figure
is_t = (XGB_model.predict_proba(np.array([[Histopathological_type,AFP,CEA,CA_125,CA_199,CA_153,CYFRA21_1,TPSA]]))[0][1])> sp
prob = (XGB_model.predict_proba(np.array([[Histopathological_type,AFP,CEA,CA_125,CA_199,CA_153,CYFRA21_1,TPSA]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk Ocular metastasis'
else:
    result = 'Low Risk Ocular metastasis'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Ocular metastasis':
        st.balloons()
    st.markdown('## Probability of High Risk Ocular metastasis group:  '+str(prob)+'%')
    

