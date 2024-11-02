import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# Load data in smaller chunks
@st.cache_data
def load_data_in_chunks(file_path, chunksize=500, sample_size=50):
    chunk_list = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        if len(chunk) > sample_size:
            chunk_list.append(chunk.sample(n=sample_size, random_state=42))
        else:
            chunk_list.append(chunk)
    return pd.concat(chunk_list, ignore_index=True)


# Plotting functions with explicit figure handling
def plot_count_reviews_by_star_rating(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    data['stars'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title('Count of Reviews by Star Rating')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Count of Reviews')
    st.pyplot(fig)


def plot_count_restaurants_by_cuisine(business_data):
    fig, ax = plt.subplots(figsize=(10, 5))
    business_data['categories'].str.split(', ').explode().value_counts().head(20).plot(kind='bar', ax=ax)
    ax.set_title('Count of Restaurants by Cuisine Type')
    ax.set_xlabel('Cuisine Type')
    ax.set_ylabel('Count of Restaurants')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


def plot_average_ratings_by_cuisine(data, business_data):
    merged_data = data.merge(business_data[['business_id', 'categories']], on='business_id')
    avg_ratings = merged_data.groupby('categories')['stars'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    avg_ratings.plot(kind='bar', ax=ax)
    ax.set_title('Average Ratings by Cuisine Type')
    ax.set_xlabel('Cuisine Type')
    ax.set_ylabel('Average Rating')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


def plot_sentiment_distribution_by_cuisine(data, business_data):
    merged_data = data.merge(business_data[['business_id', 'categories']], on='business_id')

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='stars', hue='categories', data=merged_data, ax=ax)
    ax.set_title('Sentiment Distribution by Cuisine Type')
    ax.set_xlabel('Star Rating')
    ax.set_ylabel('Count of Reviews')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Cuisine Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)


def plot_review_word_count_distribution(data):
    word_counts = data['text'].str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 5))
    word_counts.hist(bins=30, ax=ax)
    ax.set_title('Review Word Count Distribution')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)


def plot_top_positive_negative_words(data):
    all_words = ' '.join(data['text']).lower().split()
    word_counts = Counter(all_words)

    positive_words = [word for word in word_counts if
                      word in ["great", "good", "excellent", "amazing", "fantastic", "love", "best"]]
    negative_words = [word for word in word_counts if word in ["bad", "terrible", "horrible", "awful", "hate", "worst"]]

    top_positive = Counter({word: word_counts[word] for word in positive_words}).most_common(10)
    top_negative = Counter({word: word_counts[word] for word in negative_words}).most_common(10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(*zip(*top_positive))
    axes[0].set_title('Top Positive Words')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    axes[1].bar(*zip(*top_negative))
    axes[1].set_title('Top Negative Words')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    st.pyplot(fig)


# Streamlit UI
def main():
    st.title("Yelp Dataset Visualizations")

    reviews_file = "C:\\Users\\ASUS\\Downloads\\archive\\yelp_academic_dataset_review.json"
    business_file = "C:\\Users\\ASUS\\Downloads\\archive\\yelp_academic_dataset_business.json"

    st.write("Loading data...")
    reviews = load_data_in_chunks(reviews_file)
    businesses = load_data_in_chunks(business_file)

    option = st.selectbox('Select a visualization', [
        'Count of Reviews by Star Rating',
        'Count of Restaurants by Cuisine Type',
        'Average Ratings by Cuisine Type',
        'Sentiment Distribution by Cuisine Type',
        'Review Word Count Distribution',
        'Top Positive Words in Reviews',
        'Top Negative Words in Reviews'
    ])

    if option == 'Count of Reviews by Star Rating':
        plot_count_reviews_by_star_rating(reviews)
    elif option == 'Count of Restaurants by Cuisine Type':
        plot_count_restaurants_by_cuisine(businesses)
    elif option == 'Average Ratings by Cuisine Type':
        plot_average_ratings_by_cuisine(reviews, businesses)
    elif option == 'Sentiment Distribution by Cuisine Type':
        plot_sentiment_distribution_by_cuisine(reviews, businesses)
    elif option == 'Review Word Count Distribution':
        plot_review_word_count_distribution(reviews)
    elif option == 'Top Positive and Negative Words in Reviews':
        plot_top_positive_negative_words(reviews)


if __name__ == '__main__':
    main()
