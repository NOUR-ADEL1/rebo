import pandas as pd
import streamlit as st
import joblib

def load():
    leplatform=joblib.load(r'C:\Users\name\OneDrive - Benha University (Faculty Of Computers & Information Technolgy)\Desktop\power of social media\src\leplatform.pkl')
    lesentiment_score=joblib.load(r'C:\Users\name\OneDrive - Benha University (Faculty Of Computers & Information Technolgy)\Desktop\power of social media\src\lesentiment_score.pkl')
    lpost_type=joblib.load(r'C:\Users\name\OneDrive - Benha University (Faculty Of Computers & Information Technolgy)\Desktop\power of social media\src\lpost_type.pkl')
    rf=joblib.load(r'C:\Users\name\OneDrive - Benha University (Faculty Of Computers & Information Technolgy)\Desktop\power of social media\src\rf.pkl')
    return leplatform,lpost_type,lesentiment_score,rf

def main():
    st.title(' Predict likes of post ')
    leplatform,lpost_type,lesentiment_score,rf=load()
    #take user input
    input_data={}
    input_data['platform']=st.selectbox('choose plate form',['Instagram','Twitter','Facebook'])
    input_data['post_type']=st.selectbox('choose post type ',['poll','carousel','text','image','video'])
    input_data['comments']=st.number_input('enter number of comments',min_value=0,value=10,step=1)
    input_data['shares']=st.number_input('enter number of shares',min_value=0,value=10,step=1)
    input_data['sentiment_score']=st.selectbox('choose sentiment score',['neutral','negative','positive'])

    input_df=pd.DataFrame([input_data])
    input_df['platform']=leplatform.transform(input_df['platform'])
    input_df['post_type']=lpost_type.transform(input_df['post_type'])
    input_df['sentiment_score']=lesentiment_score.transform(input_df['sentiment_score'])







    if st.button('predict number of likes '):
        prediction =rf.predict(input_df)
        st.success(f'prediction number likes is:{prediction[0]}')
if __name__ == '__main__':
    main()
