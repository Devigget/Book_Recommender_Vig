import pickle
import streamlit as st
import numpy as np
import random

st.set_page_config(layout="wide")  # Set the layout to wide for better UI

st.header("Books Recommender System using Machine Learning")

# Load the pickled files (ensure that you saved these files correctly after training)
model = pickle.load(open('artifacts/model2(1).pkl', 'rb'))
book_pivot_imputed_df = pickle.load(open('artifacts/book_pivot3(1).pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating2(1).pkl', 'rb'))

books_name = book_pivot_imputed_df.index.tolist()

# Custom CSS to improve layout and spacing
st.markdown(
    """
    <style>
    .stText {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
    .stImage {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to fetch poster URLs for the recommended books
def fetch_poster(suggestions):
    book_names = []
    ids_index = []
    poster_urls = []

    # For each suggested book, fetch its name and poster URL
    for book_id in suggestions:
        book_name = book_pivot_imputed_df.index[book_id]
        book_names.append(book_name)

        # Find the corresponding book ID in final_rating to fetch the image URL
        ids = np.where(final_rating['title'] == book_name)[0]
        if len(ids) > 0:
            ids_index.append(ids[0])

    # Fetch poster URLs for the books found
    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        poster_urls.append(url)

    return poster_urls

def recommend_book(selected_book_name):
    book_list = []
    
    # Ensure the book is in the pivot table
    if selected_book_name in book_pivot_imputed_df.index:
        book_id = np.where(book_pivot_imputed_df.index == selected_book_name)[0][0]

        # Get more neighbors to allow random selection (e.g., 15 neighbors)
        distances, suggestions = model.kneighbors(book_pivot_imputed_df.iloc[book_id, :].values.reshape(1, -1), n_neighbors=15)

        # Randomize and select 5 different suggestions to ensure variety
        top_recommendations = suggestions[0][1:]  # Skip the first (input book itself)
        random.shuffle(top_recommendations)  # Shuffle the recommendations
        top_recommendations = top_recommendations[:5]  # Select top 5 after shuffling

        # Fetch poster URLs for the suggested books
        poster_urls = fetch_poster(top_recommendations)

        for i in range(len(top_recommendations)):
            book_list.append(book_pivot_imputed_df.index[top_recommendations[i]])

        return book_list, poster_urls
    else:
        st.error("Book not found in the database.")
        return [], []


# User input to select a book from the dropdown menu
selected_book = st.selectbox("Type or select a book", books_name)

# Show recommendations when the button is clicked
if st.button("Show Recommendation"):
    recommended_books, poster_urls = recommend_book(selected_book)

    # Check if we got valid recommendations
    if len(recommended_books) > 0:
        # Ensure there are enough recommendations (up to 5)
        num_recommendations = min(5, len(recommended_books), len(poster_urls))
        cols = st.columns(num_recommendations, gap="large")  # Adjust column gap for spacing

        # Display recommendations dynamically
        for i in range(num_recommendations):
            with cols[i]:
                st.image(poster_urls[i], use_column_width=True, caption=recommended_books[i])  # Display book image with title as caption
    else:
        st.text("No recommendations available.")
