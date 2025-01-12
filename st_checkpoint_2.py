import streamlit as st
import pandas as pd
import numpy as np
from  ydata_profiling import ProfileReport
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy import stats
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def app_checkpoint_2():
 st.write('##Financial Inclusion Stramlit Checkpoint II')
 df = pd.read_csv(r'C:\Users\Hp\Desktop\dataset\Streamlit\Financial_inclusion_dataset.csv')
 st.write(df.head())

 le = LabelEncoder()
 df_encoded = df.copy()
 df_encoded['country'] = le.fit_transform(df_encoded['country'])
 df_encoded['uniqueid'] = le.fit_transform(df_encoded['uniqueid'])
 df_encoded['bank_account'] = le.fit_transform(df_encoded['bank_account'])
 df_encoded['location_type'] = le.fit_transform(df_encoded['location_type'])
 df_encoded['cellphone_access'] = le.fit_transform(df_encoded['cellphone_access'])
 df_encoded['gender_of_respondent'] = le.fit_transform(df_encoded['gender_of_respondent'])
 df_encoded['relationship_with_head'] = le.fit_transform(df_encoded['relationship_with_head'])
 df_encoded['marital_status'] = le.fit_transform(df_encoded['marital_status'])
 df_encoded['education_level'] = le.fit_transform(df_encoded['education_level'])
 df_encoded['job_type'] = le.fit_transform(df_encoded['job_type'])


 X = df_encoded.drop(columns=['bank_account'])
 y = df_encoded['bank_account']
 X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)
 #X_train.shape, X_test.shape, y_train.shape, y_test.shape
 st.title('Data Encoded : ')
 st.write(df_encoded)
 st.title('Training data :')
 st.write('X_train :', X_train.shape, 'X_test :', X_test.shape)
 st.write('y_train :', y_train.shape, 'y_test :', y_test.shape)

 algorithm = st.selectbox('Choose Algorithm',
                                    options= ['Decision Tree', 'Random Forest', 'K-Nearest Neighbors', 'Grid Search'])
        
 if algorithm == 'Decision Tree':
    criterion = st.radio('Type of Criterion', options=['gini', 'entropy'])
    max_depth = st.slider('Maximum Depth', min_value=5, max_value=20, value=5)
    min_samples_leaf = st.slider('Minimum Samples Leaf', 10,50,20)
    cls = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
 elif algorithm == 'Random Forest':
    n_estimators = st.slider('Number of Estimators', min_value=10, max_value=50, value=10)
    max_depth = st.slider('Maximum Depth', min_value=2, max_value=20, value=5)
    cls = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
 elif algorithm == 'K-Nearest Neighbors' :
    n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=50, value=4)
    metric = st.selectbox('Metric', options=['euclidean', 'minkowski', 'manhattan'])
    cls = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
 elif algorithm == 'Grid Search' :
    model_grid_search = DecisionTreeClassifier()
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth' : [None,10,20,30],
              'min_samples_split' : [2,5,10]}
    cls = GridSearchCV(estimator= model_grid_search, param_grid=param_grid, cv=5, scoring='accuracy')
    
 if st.button('Train Model'):
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    tab1, tab2 = st.tabs(["Metrics", "Confusion Matrix"])

    with tab1:
        st.write("### Performence Metrics")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=2))
            
    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        fig_cm , ax = plt.subplots()
        vis = ConfusionMatrixDisplay(cm, display_labels=cls.classes_)
        vis.plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig_cm)

 else:
    st.write('No Data Found. Please upload a data on the EDA page first.')





app_checkpoint_2()