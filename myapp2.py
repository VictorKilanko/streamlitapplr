import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from math import sqrt

#streamlit dashboard made thanks to Jessie of JCharisTech- https://blog.jcharistech.com/2019/12/25/building-a-drag-drop-machine-learning-app-with-streamlit/

    
    
#import my data
graph_data = "C:\\Users\\victo\\OneDrive\\Documents\\Python Scripts\\police_hr1.csv"
graph_data=pd.read_csv(graph_data)

ml_data = "C:\\Users\\victo\\OneDrive\\Documents\\Python Scripts\\police_hr2.csv"
ml_data=pd.read_csv(ml_data)

sns.set(rc={'figure.figsize':(20,10)})

        #creating containers to organize my work
header = st.beta_container()
dataset = st.beta_container()
eda = st.beta_container()
model_training = st.beta_container()

with header:
    st.title('Texas Police Salary Dataset')
    st.text('In this project, I tried to predict salaries of Texas police based on factors like experience, age, race, sex, job title')
            #create a menu
    menu = ["View Dataset","Explore Dataset","Graph Dataset","Model Dataset","About Me"]
    choice = st.selectbox("Select a Menu",menu)


    if choice == 'View Dataset':

        st.header('Dataset for our graphs')
        st.subheader('This will be the dataset we explore later')
        st.write(graph_data.head())
        st.header('Dataset for our machine learning algorithm')
        st.write(ml_data.head())


    elif choice == 'Explore Dataset':
        st.header('Now, let us explore our graph dataset')
        st.markdown('I try to make this as interactive as possible for you to enjoy exploring insights from the data')

        st.subheader('Use the checkbox to navigate')
        if st.checkbox("Show Shape"):
            st.write(graph_data.shape)

        if st.checkbox("Show Columns"):
            st.write(graph_data.columns)

        if st.checkbox("Summary"):
            st.write(graph_data.describe())

        if st.checkbox("Show Some Selected Columns"):
            selected_columns = st.multiselect("Select Columns",graph_data.columns)
            new_df = graph_data[selected_columns]
            st.dataframe(new_df)
            
        if st.checkbox("Show Salary Distribution"):
            x = graph_data['Annual_salary'].values
            sns.boxplot(x)
            st.pyplot()
            st.write('It seems we have some outliers')
            
        if st.checkbox("See the salary outliers"):
            sns.lineplot(x="Job_title", y="Annual_salary", data=graph_data)
            st.pyplot()
            st.write('It seems our 3 directors are the salary outliers')
            
        if st.checkbox("Show Job Title by Experience"):
            st.markdown('Are our directors the most experienced or why are they paid more?')
            sns.barplot(x="Job_title", y="Experience_years", data=graph_data)
            st.pyplot()
            
        if st.checkbox("Exploring if directors have the most experience"):
            u = pd.DataFrame(graph_data[['Job_title', 'Experience_years']].groupby('Job_title', as_index = True).max())
            st.write('This table shows the maximum years of experience for each job title. So, our directors are not the most experienced.')
            st.dataframe(u)
            st.bar_chart(data=u)
            st.pyplot()
            
        if st.checkbox("Show Age Distribution"):
            x = graph_data['Age'].values
            sns.boxplot(x)
            st.pyplot()
            
                #if st.checkbox("Show Value Counts"):
                    #st.write(graph_data.iloc[:,-1].value_counts())

    elif choice == 'Graph Dataset':      
        st.header('Now, let us graph our dataset')
        st.markdown('I try to make this as interactive as possible for you to enjoy exploring insights from the data')
        
        #let's create columns for the graphs
        col_1,col_2 = st.beta_columns(2)
        
        col_2.header('To customize your own graph')
        graph_data= graph_data[['Annual_salary','Race','Sex','Age','Experience_years','Job_title']]
        graph_data['Annual_salary']=graph_data['Annual_salary'].astype(int)
        columns_names = graph_data.columns.tolist()
        
        #columns_names = columns_names.apply(lambda x:x[1:])
        #columns_names = columns_names.keep('Annual_salary','Race','Sex','Age','Experience_years','Job_title',axis=1).values
        #columns_names = pd.DataFrame(graph_data.columns.tolist())
        #columns_names = columns_names.astype('int')
        col_2.write('The columns we have are {}'.format(columns_names))      
        plotname = col_2.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
                #selected_columns_names = st.multiselect("Select Columns To Plot",columns_names)
        Y_variable = col_2.selectbox("Select your Y variable",columns_names)
        X_variable = col_2.selectbox("Select your X variable",columns_names)

        if col_2.button("Generate Plot"):
            col_2.success("Generating Customizable Plot of {} vs {}".format(X_variable,Y_variable))
                #my plot
        if plotname =='bar':
            sns.barplot(x=X_variable, y=Y_variable, data=graph_data)
            col_2.pyplot()
            
        if plotname =='hist':
            col_2.markdown('Just select an X for this')
            graph_data[X_variable].astype(float)
            sns.distplot(x=X_variable)
            col_2.pyplot()
            
            
        col_1.header('Here are some pre-made graphs')
        
        col_1.subheader('A bar chart of Age vs Salary')
        sns.barplot(x="Age", y="Annual_salary", data=graph_data)
        col_1.pyplot()
        
        col_1.subheader('A bar chart of Age vs Salary vs Sex')
        sns.barplot(x="Age", y="Annual_salary", hue='Sex', data=graph_data)
        col_1.pyplot()
        
        col_1.subheader('A bar chart of Experience vs Job title')
        sns.barplot(x="Experience_years", y="Job_title", data=graph_data)
        col_1.pyplot()
        
        col_1.subheader('A bar chart of Job title vs Salary')
        sns.barplot(x="Job_title", y="Annual_salary", data=graph_data)
        col_1.pyplot()
        
        col_1.subheader('A bar chart of Job title vs Salary vs Sex')
        sns.barplot(x="Job_title", y="Annual_salary", hue='Sex',data=graph_data, ci=False)
        col_1.pyplot()
        
        col_1.subheader('A bar chart of Job title vs Salary vs Race')
        sns.barplot(x="Job_title", y="Annual_salary", hue='Race',data=graph_data, ci=False)
        col_1.pyplot()

