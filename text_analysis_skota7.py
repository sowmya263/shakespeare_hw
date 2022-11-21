from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Importing the StringIO module.
from io import StringIO 
import nltk
import nltk
nltk.download('punkt')
from collections import Counter
import seaborn as sns


st.write('## Analyzing Shakespeare texts')

st.sidebar.header("Word Cloud Settings")
max_word = st.sidebar.slider("Max Words",10,200,100,10)
max_font = st.sidebar.slider("Size of Largest Word",50,350,60)
image_size = st.sidebar.slider("image Width",100,800,400,10)
random = st.sidebar.slider("Random State",30,100,42)

remove_stopwords = st.sidebar.checkbox('Remove stop words?',value = True)

st.sidebar.header("Word Count Settings")
word_count = st.sidebar.slider("Minimun Count of Words",5,100,40)



 # image = st.file_uploader("Choose a txt file")
books = {" ":" ","A Mid Summer Night's Dream":"data/summer.txt",
         "The Merchant of Venice":"data/merchant.txt","Romeo and Juliet":"data/romeo.txt"}

final = st.selectbox("Choose a txt file",books)
image = books[final]

if image is not " ":
    raw_text = open(image,"r").read().lower()
    with open(image) as file:
    # To read file as string:
      dataset = file.read()
    
    stopwords = set(STOPWORDS)
    stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
    'little', 'say', 'must', 'way', 'long', 'yet', 'mean',
    'put', 'seem', 'asked', 'made', 'half', 'much',
    'certainly', 'might', 'came','o'])
    
tab1, tab2, tab3 = st.tabs(['Word Cloud','Bar Chart','View Text'])

with tab1:
    if image is not " ":
        if remove_stopwords: 
           cloud = WordCloud(background_color = "white", 
                            max_words = max_word, 
                            max_font_size=max_font, 
                            stopwords = stopwords, 
                            random_state=random)
        else:
            cloud = WordCloud(background_color = "white", 
                            max_words = max_word, 
                            max_font_size=max_font, 
                            random_state=random)
        wc = cloud.generate(dataset)
        word_cloud = cloud.to_file('wordcloud.png')  
        st.image(wc.to_array(), width = image_size)

with tab2:
    if image is not " ":
        
        st.write('Bar chart')
            
        tokens = nltk.word_tokenize(dataset)
        tokens = [t for t in tokens if t.isalpha()]
        sw_remove = [w for w in tokens if not w.lower() in stopwords]
        if remove_stopwords:
            frequency = nltk.FreqDist(sw_remove)
            freq_df = pd.DataFrame(frequency.items(),columns=['word','count'])
            sorted_data = freq_df.sort_values("count", ascending=False)
            df = sorted_data[ sorted_data.iloc[:,1]>= word_count ]
            bars = alt.Chart(df).mark_bar().encode(
                x='count',
                y=alt.Y('word:N', sort='-x')
            )
            st.altair_chart(bars, use_container_width=True)
        
        else:
            frequency = nltk.FreqDist(tokens)
            freq_df = pd.DataFrame(frequency.items(),columns=['word','count'])
            sorted_data = freq_df.sort_values("count", ascending=False)
            df = sorted_data[ sorted_data.iloc[:,1]>= word_count ]
             
            bars = alt.Chart(df).mark_bar().encode(
                x='count',
                y=alt.Y('word:N', sort='-x')
            )
            st.altair_chart(bars, use_container_width=True)
        

with tab3:
    if image is not " ":
        st.write(dataset)           
