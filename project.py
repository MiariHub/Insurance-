import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 
import plost
import os
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
color_palette = ["#7456A3","#08BEDF","#008DC3"]
sns.set_style("whitegrid" )
sns.set_theme(style="ticks")
sns.set_palette(color_palette)
import pickle
from sklearn.ensemble import  RandomForestClassifier
with open('GBR.pkl', 'rb') as file:
    loaded_data = pickle.load(file)


centered_style = """
    display: flex;
    justify-content: center;
    align-items: center;
    """


footer_style = """
position: fixed;
bottom: 0;
width: 100%;
background-color: #07b3d3;
text-align: center;
padding: 10px;
"""

page_bg = """

    background-color: rgb(3 119 172)
"""


st.header("Prediction for Insurance using AI")


tab1, tab2, tab3 = st.tabs(["Home", "Dashboard", "Prediction"])



with tab1:
    # Display an image on the main page
    image = Image.open('in.png')
    

    col1, col2= st.columns([1,1])
    with col1:
        st.image(image, caption=None, width=400, use_column_width=None, clamp=False, channels="RGB", output_format="auto", )


    
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown('<div style="text-align: Justify;"> Welcome to our innovative platform where we harness the power of AI to transform medical insurance. Our machine learning model, Gradient Boosting, analyzes diverse factors such as age, BMI, smoking status, and more to uncover deep insights into healthcare costs.

Our approach allows us to offer personalized insurance plans tailored to individual health profiles, encouraging proactive health management and potentially reducing premiums. 

Explore how our AI-driven tools are setting new standards in medical insurance, making policies more affordable and aligned with your unique health needs.</div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div style="text-align: Right;">By Developer.</div>', unsafe_allow_html=True)

with tab2:
    df = pd.read_csv('insurance.csv', index_col=0)
    # Add title to the dashboard
    st.title('Dataset Dashboard')

    # Display the dataset
    st.write('## Dataset Overview')       
    st.write(df)

    #visualizations
    st.write('## Data Visualizations')

    # Example: Bar chart to visualize distribution of a numerical variable
    st.subheader('1.Sex')
    plt.figure(figsize=(15,8))
    fig = sns.catplot(data=df, x="sex", kind="count",palette=color_palette)
    plt.title('The Count of Male & Female')
    plt.ylabel('Count')
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('2.Smoker')
    plt.figure(figsize=(15,8))
    fig = sns.catplot(data=df, x="smoker", kind="count",palette=color_palette)
    plt.title('The Count of Smoker')
    plt.ylabel('Count')
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('3.region')
    plt.figure(figsize=(15,8))
    fig = sns.catplot(data=df, x="region", kind="count",palette=color_palette)
    plt.title('The Count of Region')
    plt.ylabel('Count')
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('4.Children')
    plt.figure(figsize=(15,8))
    fig = sns.catplot(data=df, x="children", kind="count",palette=color_palette)
    plt.title('The Count of Childern')
    plt.ylabel('Count')
    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('Pivots')
    pivot = pd.pivot_table(df, index = ['region','sex','smoker'],aggfunc=np.mean)
    pivot

with tab3:
    with st.form("prediction_form"):
        st.subheader("General Information")
        c1,c2 = st.columns(2)
        sex = c1.radio('Gender', ['Male', 'Female'])
        region = c2.radio('Region', ['Northeast', 'Northwest', 'Southeast', 'Southwest'])
        c1,c2 = st.columns(2)
        children = c1.selectbox('Children', list(range(6)))
        smoker = c2.selectbox('Do you smoke?', ['Yes', 'No'])
        age = st.slider('Age', 19, 110, 25)
        bmi = st.slider('BMI', 16.0, 50.0, 22.0)
        submit_button = st.form_submit_button("Predict")

        if submit_button:
            # Convert inputs to match model training format
            sex = 1 if sex == 'Male' else 0
            smoker = 0 if smoker == 'Yes' else 1
            region_map = {'Northeast': 1, 'Northwest': 2, 'Southeast': 3, 'Southwest': 4}
            region = region_map[region]

            # Prepare input data for prediction
            input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                                      columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            prediction = loaded_data.predict(input_data)
            st.success(f'Predicted Insurance Cost: ${prediction[0]:.2f}')
    
    
    
            


if tab1 == "Home":
        Home()
elif tab2 == "Dashboard":
        Dashboard()
elif tab3 == "Prediction":
        Prediction()
    



def main():
   

    # Use the HTML component to add a fixed footer
    st.markdown('<div style="{}">Prdiction Issurance using AI.</div>'.format(footer_style), unsafe_allow_html=True)

if __name__ == "__main__":
    main()









