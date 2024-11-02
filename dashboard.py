import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import nltk
from nltk import ngrams
from collections import Counter

# Download NLTK data files (only need to run once)
#nltk.download('punkt')
#nltk.download('punkt')
from helper_functions import *



import warnings
warnings.filterwarnings("ignore")

####### Load Dataset ###########################################################

df = pd.read_csv('processed_data.csv', usecols=['rating','text','domain','clean_text'] ,delimiter=',')
x=df['text']
y=df['rating']

#########################Page Title###############################

#st.set_page_config(page_title="Analysis App",page_icon="📊")

# app design
app_meta()
set_bg_hack('istockphoto-1363652676-1024x1024.jpg')
html_temp = """
  <div style="background-color: rgba(0, 0, 0, 0.7); 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8);
                text-align: center;">
        <h1 style="color: white; font-size: 50px; margin: 0; 
                   text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);">Multi-Domain Sentiment Analysis</h1>
        <p style="color: white; font-size: 20px; margin: 10px 0; 
                   text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.8);">Discover insights and trends from customer reviews</p>
    </div>	"""
st.markdown(html_temp, unsafe_allow_html=True)

# Center-align the main title using HTML
st.markdown(
    "<h2 style='text-align: center; color: white;'>Customer Reviews EDA 📊</h2>", 
    unsafe_allow_html=True
)


######
 #Write Summary of the Tweets
# Write Summary of the Tweets

#st.markdown("<p style='color: white;'>Total Reviews are: {}</p>".format(len(df)), unsafe_allow_html=True)
#st.markdown("<p style='color: white;'>Total Positive Reviews are: {}</p>".format(len(df[df["rating"] == 2])), unsafe_allow_html=True)
#st.markdown("<p style='color: white;'>Total Negative Reviews are: {}</p>".format(len(df[df["rating"] == 0])), unsafe_allow_html=True)
#st.markdown("<p style='color: white;'>Total Neutral Reviews are: {}</p>".format(len(df[df["rating"] == 1])), unsafe_allow_html=True)

# Write Summary of the Tweets as Metrics
# Create a container div for the metrics
# Development Metrics Section
st.markdown("<h2 style='color: white; text-align: center;'>Reviews Metrics</h2>", unsafe_allow_html=True)

# Create columns for metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Total metrics
total_reviews = len(df)
total_positive_reviews = len(df[df["rating"] == 2])
total_negative_reviews = len(df[df["rating"] == 0])
total_neutral_reviews = len(df[df["rating"] == 1])

# Display metrics in the columns
# Display metrics in the columns
with kpi1:
    st.markdown(
        "<div style='background-color: rgba(255, 255, 255, 0.9); "
        "padding: 0px; border-radius: 0px; "
        "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
        "<h3 style='color: black; font-size: 20px; margin: 0;'>Total Reviews</h3>"
        "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{}</h2></div>".format(total_reviews), 
        unsafe_allow_html=True
    )

with kpi2:
    st.markdown(
        "<div style='background-color: rgba(255, 255, 255, 0.9); "
        "padding: 0px; border-radius: 0px; "
        "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
        "<h3 style='color: black; font-size: 20px; margin: 0;'>Total Positive Reviews</h3>"
        "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{}</h2></div>".format(total_positive_reviews), 
        unsafe_allow_html=True
    )

with kpi3:
    st.markdown(
        "<div style='background-color: rgba(255, 255, 255, 0.9); "
        "padding: 0px; border-radius: 0px; "
        "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
        "<h3 style='color: black; font-size: 20px; margin: 0;'>Total Negative Reviews</h3>"
        "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{}</h2></div>".format(total_negative_reviews), 
        unsafe_allow_html=True
    )

with kpi4:
    st.markdown(
        "<div style='background-color: rgba(255, 255, 255, 0.9); "
        "padding: 0px; border-radius: 0px; "
        "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
        "<h3 style='color: black; font-size: 20px; margin: 0;'>Total Neutral Reviews</h3>"
        "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{}</h2></div>".format(total_neutral_reviews), 
        unsafe_allow_html=True
    )


## filteration method #############
# Fill any NaN values and convert 'clean_text' to string type
# Calculate word and character counts
df['word_count'] = df['clean_text'].fillna('').astype(str).apply(lambda x: len(x.split(" ")))
df['char_count'] = df['clean_text'].fillna('').astype(str).apply(lambda x: len(x))

# Map each review type to its corresponding rating value
rating_map = {
    'Positive Reviews': 2,
    'Neutral Reviews': 1,
    'Negative Reviews': 0,
    'All Reviews' : 3
}


# Streamlit selectbox
topic = st.selectbox('Select review type:', ['All Reviews','Positive Reviews', 'Neutral Reviews', 'Negative Reviews'])

# Filter the data based on selected review type
selected_rating = rating_map[topic]
filtered_df = df[df['rating'] == selected_rating]
################# Scatter Chart Logic #################

st.sidebar.markdown("<h2 style='text-align: center;'>Pages 📊</h2>", unsafe_allow_html=True)

st.sidebar.header("About App")
st.sidebar.info("A Customer Sentiment analysis Project which collect data of reviews of different domains. The reviews will then be used to determine the Sentiments of those reviews. \
                The different Visualizations will help us get a feel of the overall exploration of reviews")
st.sidebar.text("Built with Streamlit")

st.sidebar.header("For Any Queries/Suggestions Please reach out at :")
st.sidebar.info("medotalaat20177@gmail.com")

################# Wordcloud #################

# Define rating mapping
def get_reviews_by_rating(df, rating):
    if rating ==3 :
        reviews = df['clean_text'].str.cat(sep=' ')
    else :   
        reviews = df[df['rating'] == rating]['clean_text'].str.cat(sep=' ')
    return reviews

# Create word cloud based on selected rating
def create_wordcloud(topic_rating):
    if   topic_rating == 'Positive Reviews':
        reviews_text = get_reviews_by_rating(df, 2.0)
    elif topic_rating == 'Neutral Reviews':
        reviews_text = get_reviews_by_rating(df, 1.0)
    elif topic_rating == 'Negative Reviews':
        reviews_text = get_reviews_by_rating(df, 0.0)
    else : 
        reviews_text = get_reviews_by_rating(df,3)    

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(reviews_text)
    return wordcloud


# Generate word cloud based on the selection
if topic in ['All Reviews'] :
    wordcloud_all = create_wordcloud(topic)
    # Display the word cloud
    wordcloud_fig_all, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud_all, interpolation='bilinear')
    ax.set_title("All Reviews",fontsize=16)
    ax.axis("off")

    wordcloud_pos = create_wordcloud('Positive Reviews')
    # Display the word cloud
    wordcloud_fig_pos, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud_pos, interpolation='bilinear')
    ax.set_title("Positive Reviews",fontsize=16)
    ax.axis("off")

    wordcloud_neu = create_wordcloud('Neutral Reviews')
    # Display the word cloud
    wordcloud_fig_neu, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud_neu, interpolation='bicubic')
    ax.set_title("Neutral Reviews",fontsize=16)
    ax.axis("off")

    wordcloud_neg = create_wordcloud('Negative Reviews')
    # Display the word cloud
    wordcloud_fig_neg, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud_neg, interpolation='bicubic')
    ax.set_title("Negative Reviews",fontsize=16)
    ax.axis("off")
else:
    wordcloud = create_wordcloud(topic)
    # Display the word cloud
    wordcloud_fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    #st.pyplot(wordcloud_fig)
#####################Distribution Plot######################################################



# Display filtered data and Generate distribution plot for the filtered and all the data     


if topic in ['All Reviews'] and selected_rating ==3:
    st.write(f"All Data", df)
    fig_dist = ff.create_distplot(
        [df[df['rating'] == 2]['word_count'], df[df['rating'] == 1]['word_count'],df[df['rating'] == 0]['word_count'],df['word_count']],
        ['Positive','Neutral', 'Negative',topic],
        bin_size=10,
        show_hist=False,
        show_rug=False
    )

    # Customize plot layout
    fig_dist.update_layout(
        title=f"Distribution of Words in {topic}",
        xaxis_title='No. of Words',
        yaxis_title='Frequency',
        height=500,
        width=700,
        plot_bgcolor='black',  # Change the background color of the plot
        paper_bgcolor='black',  # Change the background color of the paper
        font=dict(color='white')  # Set font color to white for better visibility
    )
    

    # Display the plot in Streamlit
    #st.plotly_chart(fig_dist)
elif topic not in ['All Reviews'] and selected_rating in [0,1,2] :
    #filtered_df = df[df['rating'] == selected_rating]
    st.write(f"Data for {topic}:", filtered_df)        
    fig_dist = ff.create_distplot(
        [filtered_df['word_count']],
        [topic],
        bin_size=10,
        show_hist=False,
        show_rug=False
    )

    # Customize plot layout
    fig_dist.update_layout(
        title=f"Distribution of Words in {topic}",
        xaxis_title='No. of Words',
        yaxis_title='Frequency',
        height=500,
        width=700
    )

    # Display the plot in Streamlit
    #st.plotly_chart(fig_dist)

####################Histogram##############################################

 #Filter data based on selected review type
if topic in ['All Reviews'] and selected_rating ==3: # All Reviews
    
    # Plot histogram for all review types combined
    fig_histogram = px.histogram(
        df, 
        x='word_count', 
        color='rating', 
        category_orders={'rating': ['Positive','Neutral', 'Negative']},
        labels={'rating': 'Review Type', 'word_count': 'Number of Words'},
        title='Distribution of Words in All Reviews',
        nbins=15,  # Adjust bins if needed
        height=600,
        width=800,
        marginal='violin'
    )
    fig_histogram.update_layout(barmode='overlay')
    fig_histogram.update_traces(opacity=0.6)
    #st.plotly_chart(fig_histogram)

elif topic not in ['All Reviews'] and selected_rating in [0,1,2] :
    # Filter the DataFrame for the selected rating only
    #filtered_df = df[df['rating'] == selected_rating]
    #st.write(f"Data for {topic}:", filtered_df)
    
    # Plot histogram for the selected review type
    fig_histogram = px.histogram(
        filtered_df,
        x='word_count',
        title=f'Distribution of Words in {topic}',
        nbins=15,
        height=600,
        width=800,
        marginal='violin'
    )
    fig_histogram.update_layout(
        xaxis_title='No. of Words',
        yaxis_title='Frequency',
        showlegend=False
    )
    #st.plotly_chart(fig_histogram)    


##########################################Histogram of Review Length ###########################################################
if topic == 'All Reviews' and selected_rating == 3:
# Create a histogram with a kernel density estimate (KDE) for all review types combined
    fig_hist_len = px.histogram(
        df,
        x='char_count',
        color='rating',
        category_orders={'rating': [2, 1, 0]},  # Assuming 2 for Positive, 1 for Neutral, 0 for Negative
        nbins=25,  # Number of bins
        title='Distribution of Review Lengths',
        labels={'char_count': 'Review Length'},
        marginal='violin',  # Add a violin plot as marginal
        height=600,
        width=800
    )
    # Update layout to improve readability
    fig_hist_len.update_layout(
        xaxis_title='Review Length',
        yaxis_title='Frequency',
        barmode='overlay',  # Overlay bars for better visibility
    )
    # Display the histogram in Streamlit
    #st.plotly_chart(fig_hist_len)

elif topic not in ['All Reviews'] and selected_rating in [0, 1, 2]:

    # Create a histogram with a kernel density estimate (KDE) for the selected review type
    fig_hist_len = px.histogram(
        filtered_df,
        x='char_count',
        nbins=25,  # Number of bins
        title=f'Distribution of Review Lengths for {topic}',
        labels={'char_count': 'Review Length'},
        marginal='violin',  # Add a violin plot as marginal
        height=600,
        width=800
    )
    # Update layout to improve readability
    fig_hist_len.update_layout(
        xaxis_title='Review Length',
        yaxis_title='Frequency',
    )
    # Display the histogram in Streamlit
    #st.plotly_chart(fig_hist_len)
###########################Pie Chart ###############################################

# Calculate counts for sentiments
if topic == 'All Reviews' and selected_rating == 3:
    # Get counts for all sentiments
    positive_count = df[df['rating'] == 2].shape[0]
    neutral_count = df[df['rating'] == 1].shape[0]
    negative_count = df[df['rating'] == 0].shape[0]
    
    # Create a pie chart for all reviews
    fig_pie = px.pie(
        values=[positive_count, negative_count, neutral_count],
        title="Distribution of Sentiments",
        names=['Positive', 'Negative', 'Neutral'],
        hover_name=['Positive', 'Negative', 'Neutral'],
        opacity=0.9,
        template="plotly_white"
    )

    # Update layout dimensions for better visibility
    fig_pie.update_layout(width=700,
        height=500,  margin=dict(l=4, r=4, t=8, b=4))  # Adjust margins for better spacing)
    # Add annotations for better readability
    fig_pie.update_traces(
    textinfo='percent+label',  # Show percentage and label on slices
    pull=[0.09, 0.09, 0.09]  # Slightly pull out slices for emphasis
    )
    # Display the pie chart
    #st.plotly_chart(fig_pie)

elif topic not in ['All Reviews'] and selected_rating in [0, 1, 2]:
    # Filter the DataFrame for the selected rating only
    #filtered_df = df[df['rating'] == selected_rating]
    
    # Calculate counts for the selected sentiment
    positive_count = filtered_df[filtered_df['rating'] == 2].shape[0]
    neutral_count = filtered_df[filtered_df['rating'] == 1].shape[0]
    negative_count = filtered_df[filtered_df['rating'] == 0].shape[0]
    
    # Create a pie chart for the selected review type
    fig_pie = px.pie(
        values=[positive_count, negative_count, neutral_count],
        title=f"Distribution of Sentiments for {topic}",
        names=['Positive', 'Negative', 'Neutral'],
        hover_name=['Positive', 'Negative', 'Neutral'],
        opacity=0.9,
        template="plotly_white"
    )

    # Update layout dimensions for better visibility
    fig_pie.update_layout(width=600, height=450)

    # Display the pie chart
    #st.plotly_chart(fig_pie)
########################Box Plot##################################### 
# Assuming df is your DataFrame and necessary calculations have been done
# Calculate character count for each review if not already done
# Create a histogram with a kernel density estimate (KDE) for all review types combined
if topic == 'All Reviews' and selected_rating == 3:
    # Create a box plot for review lengths
    fig_box = px.box(
        df,
        x='rating',
        y='char_count',
        title='Review Length by Sentiment',
        labels={'rating': 'Sentiment', 'char_count': 'Review Length'},
        color='rating'  # Optional: colors by rating
    )
    
    # Update layout for box plot
    fig_box.update_layout(
        width=700,
        height=500,
        xaxis_title='Sentiment',
        yaxis_title='Review Length',
    )
    
    # Display the box plot in Streamlit
    #st.plotly_chart(fig_box)

    # Calculate and display average review length
    #avg_length = df.groupby('rating')['char_count'].mean().reset_index()
    #st.write("Average review length by sentiment:")
    #st.dataframe(avg_length)
    # Calculate average review length
    avg_length_positive = df[df['rating'] == 2]['char_count'].mean()
    avg_length_neutral = df[df['rating'] == 1]['char_count'].mean()
    avg_length_negative = df[df['rating'] == 0]['char_count'].mean()

    # Display average review lengths without DataFrame
    # Display average review lengths without DataFrame
    st.markdown("<h2 style='color: white; text-align: center;'>Average Review Lengths</h2>", unsafe_allow_html=True)

    # Create columns for metrics
    length_kpi1, length_kpi2, length_kpi3 = st.columns(3)

    # Display average review lengths with the new styling
    with length_kpi1:
        st.markdown(
            "<div style='text-align: center; background-color: rgba(255, 255, 255, 0.9); "
            "padding: 0px; border-radius: 0px; "
            "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
            "<h3 style='color: black; font-size: 20px; margin: 0;'>Positive Reviews</h3>"
            "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{:.2f} characters</h2></div>".format(avg_length_positive), 
            unsafe_allow_html=True
        )

    with length_kpi2:
        st.markdown(
            "<div style='text-align: center; background-color: rgba(255, 255, 255, 0.9); "
            "padding: 0px; border-radius: 0px; "
            "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
            "<h3 style='color: black; font-size: 20px; margin: 0;'>Neutral Reviews</h3>"
            "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{:.2f} characters</h2></div>".format(avg_length_neutral), 
            unsafe_allow_html=True
        )

    with length_kpi3:
        st.markdown(
            "<div style='text-align: center; background-color: rgba(255, 255, 255, 0.9); "
            "padding: 0px; border-radius: 0px; "
            "box-shadow: 0 10px 50px rgba(0, 0, 0, 0.8); text-align: center;'>"
            "<h3 style='color: black; font-size: 20px; margin: 0;'>Negative Reviews</h3>"
            "<h2 style='color: black; font-size: 30px; margin: 10px 0;'>{:.2f} characters</h2></div>".format(avg_length_negative), 
            unsafe_allow_html=True
        )


elif topic not in ['All Reviews'] and selected_rating in [0, 1, 2]:
     # Create a box plot for the filtered data
    fig_box = px.box(
        filtered_df,
        x='rating',
        y='char_count',
        title='Review Length by Sentiment',
        labels={'rating': 'Sentiment', 'char_count': 'Review Length'},
        color='rating'
    )
    
    # Update layout for box plot
    fig_box.update_layout(
        width=700,
        height=500,
        xaxis_title='Sentiment',
        yaxis_title='Review Length',
    )
    
    # Display the box plot in Streamlit
    #st.plotly_chart(fig_box)

    # Calculate and display average review length for the filtered data
    #avg_length_filtered = filtered_df.groupby('rating')['char_count'].mean().reset_index()
    #st.write("Average review length by sentiment for selected reviews:")
    #st.dataframe(avg_length_filtered)
   # Calculate average review length for the filtered data
    avg_length = filtered_df['char_count'].mean()

    # Display average review length without DataFrame
    #st.write(f"Average Review Length for {topic}:")
    #st.metric("Review Length", f"{avg_length:.2f} characters")
    # Display average review length without DataFrame
    st.markdown(f"<h2 style='color: white; text-align: center;'>Average Review Length for {topic}</h2>", unsafe_allow_html=True)

    # Create a single column for the metric
    st.markdown(
        f"<div style='text-align: center; background-color: black; padding: 20px; border-radius: 10px;'>"
        f"<h3 style='color: white;'>Review Length</h3>"
        f"<h2 style='color: white;'>{avg_length:.2f} characters</h2>"
        f"</div>",
        unsafe_allow_html=True
    )

 ##################### N- Gram #########################################


# Function to get n-grams from a given text
def get_ngrams(text, n):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Generate n-grams
    n_grams = ngrams(tokens, n)
    return n_grams

# Function to extract n-grams from the DataFrame
def extract_ngrams(df, n):
    # Flatten the list of n-grams and count frequency
    n_grams = Counter()
    for text in df['clean_text'].astype(str):
        n_grams.update(get_ngrams(text, n))
    return n_grams.most_common(10)  # Get the 10 most common n-grams

# Function to plot n-grams
def plot_ngrams(ngrams_list, title):
    # Check if ngrams_list is empty
    if not ngrams_list:
        st.write(f"No {title} found.")
        return  # Exit the function if there are no n-grams to plot
    
    # Prepare data for plotting
    labels, values = zip(*ngrams_list)
    labels = [' '.join(gram) for gram in labels]  # Join tuples to create strings

    # Create bar plot
    plt.figure(figsize=(10, 5))
    plt.barh(labels, values, color='skyblue')
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest frequency on top
    st.pyplot(plt)  # Use Streamlit to display the plot
    plt.clf()  # Clear the current figure to prevent overlap


# Assume df is your DataFrame containing the reviews with 'rating' and 'clean_text' columns
if topic in ['All Reviews'] and selected_rating == 3:
    # Extract and display frequent bigrams and trigrams
    bigrams = extract_ngrams(df, 2)
    trigrams = extract_ngrams(df, 3)

    st.subheader("Top Bigrams")
    st.write(bigrams)
    #plot_ngrams(bigrams, 'Top 10 Bigrams')

    st.subheader("Top Trigrams")
    st.write(trigrams)
    #plot_ngrams(trigrams, 'Top 10 Trigrams')


elif topic not in ['All Reviews'] and selected_rating in [0, 1, 2]:
    # Filter DataFrame for the selected rating and topic
    #filtered_df = df[(df['rating'] == selected_rating) & (df['clean_text'].str.contains(topic, case=False))]
    
    # Check if filtered DataFrame is empty
    if filtered_df.empty:
        st.write(f"No data found for {topic} with rating {selected_rating}.")
    else:
        # Extract and display frequent bigrams and trigrams
        bigrams = extract_ngrams(filtered_df, 2)
        trigrams = extract_ngrams(filtered_df, 3)

        st.subheader("Top Bigrams")
        st.write(bigrams)
        #plot_ngrams(bigrams, 'Top 10 Bigrams')

        st.subheader("Top Trigrams")
        st.write(trigrams)
        #plot_ngrams(trigrams, 'Top 10 Trigrams')

      
########## App sidebar Logic ##################

st.sidebar.markdown("### Pages ")



##################### Layout Application ##################


if topic in ['All Reviews'] and selected_rating ==3:
    container0 = st.container()
    col1, col2 = st.columns(2)
    container00 = st.container()
    col1, col2 = st.columns(2)
    with container0:
        with col1:
            st.pyplot(wordcloud_fig_all)
        with col2:
            st.pyplot(wordcloud_fig_pos)


    with container00:
        with col1:
            st.pyplot(wordcloud_fig_neu)
        with col2:
            st.pyplot(wordcloud_fig_neg)

elif topic not in ['All Reviews'] and selected_rating in [0,1,2] :   
    container4 = st.container()
    col1 = st.columns(1)[0]
    with container4:
        with col1:
            st.pyplot(wordcloud_fig)
        

       

container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        st.plotly_chart(fig_histogram)
    with col2:
        st.plotly_chart(fig_hist_len)


container2 = st.container()
col3, col4 = st.columns(2)

with container2:
    with col3:
        st.plotly_chart(fig_box)
    with col4:
        st.plotly_chart(fig_pie)    




container3 = st.container()
col5, col6 = st.columns(2)

with container3:
    with col5:
        plot_ngrams(bigrams, 'Top 10 Bigrams')
    with col6:
        plot_ngrams(trigrams, 'Top 10 Bigrams')



