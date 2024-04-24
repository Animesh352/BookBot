import streamlit as st
from openai import OpenAI
import os
from utils import text_to_embedding
from pinecone_config import init_pinecone

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
index = init_pinecone()

def query_model(query):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}]
    )
    return completion.choices[0].message.content.strip()

def get_book_metadata(user_query):
    query_vector = text_to_embedding(user_query)
    response = index.query(vector=query_vector, top_k=1, include_metadata=True)
    matches = response.get('matches', [])
    return matches[0]['metadata'] if matches else None

def get_similar_books(query_text, exclude_title):
    query_embedding = text_to_embedding(query_text)
    results = index.query(vector=query_embedding, top_k=6, include_metadata=True)
    filtered_results = [book for book in results['matches'] if book['metadata']['book_title'] != exclude_title]
    return filtered_results[:5]  # Return the top 5 books excluding the queried one

def expand_summary(summary):
    expanded = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": summary}]
    )
    return expanded.choices[0].message.content.strip()

language_dict = {
    'ar': 'Arabic', 'ca': 'Catalan', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German',
    'el': 'Greek', 'en': 'English', 'eo': 'Esperanto', 'es': 'Spanish', 'fa': 'Persian (Farsi)',
    'fr': 'French', 'ga': 'Irish', 'gd': 'Scottish Gaelic', 'gl': 'Galician', 'hi': 'Hindi',
    'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'la': 'Latin', 'ms': 'Malay',
    'nl': 'Dutch', 'no': 'Norwegian', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian',
    'ru': 'Russian', 'th': 'Thai', 'tl': 'Tagalog (Filipino)', 'vi': 'Vietnamese',
    'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Traditional)'
}

def main():
    st.sidebar.title("Current Book Details")
    user_query = st.sidebar.text_input("Search for a book:", "")
    st.title("Book Information and Recommendation Chatbot")
    tabs = st.tabs(["Recommended Books", "Chat"])

    if user_query:
        book_metadata = get_book_metadata(user_query)
        if book_metadata:
            st.sidebar.image(book_metadata['img_l'] if book_metadata['img_l'] else "default_image.jpg", use_column_width=True)
            st.sidebar.write("Title: " + book_metadata['book_title'])
            st.sidebar.write("Author: " + book_metadata['book_author'])
            
            if book_metadata['year_of_publication']:
             year_of_publication = int(float(book_metadata['year_of_publication']))
             st.sidebar.write(f"Year of Publication: {year_of_publication}")
            else:
             st.sidebar.write("Year of Publication: Unknown")

            #st.sidebar.write("Year: " + str(book_metadata['year_of_publication']))
            st.sidebar.write("Publisher: " + book_metadata['publisher'])

            full_language_name = language_dict.get(book_metadata['Language'], "Unknown Language")
            st.sidebar.write(f"Language: {full_language_name}")
            #st.sidebar.write("Language: " + book_metadata['Language'])
            #st.sidebar.write("Category: " + book_metadata['Category'])

            if book_metadata['Category']:
                # Extract the category, assuming it's formatted like "['Fiction']"
                category = book_metadata['Category']
                # Check if the string starts with '[' and ends with ']', then slice it out
                if category.startswith("[") and category.endswith("]"):
                    category = category[1:-1]  # Remove the first and last character
            else:
                category = "Unknown Category"
            st.sidebar.write(f"Category: {category}")
            
            expanded_summary = expand_summary(book_metadata['Summary'])
            st.sidebar.write("Summary: " + expanded_summary)

    with tabs[0]:

        if user_query:
            recommended_books = get_similar_books(user_query, book_metadata['book_title'] if book_metadata else "")
            if recommended_books:
                for book in recommended_books:
                    st.image(book['metadata']['img_l'] if book['metadata']['img_l'] else "default_image.jpg", caption=book['metadata']['book_title'])
                    st.write("Title: " + book['metadata']['book_title'])
                    st.write("Author: " + book['metadata']['book_author'])
                    language_name = language_dict.get(book['metadata']['Language'], "Unknown Language")
                    st.write(f"Language: {language_name}")
                    st.write("Summary: " + expand_summary(book['metadata']['Summary']))
                    st.markdown("---")

    with tabs[1]:
        chat_input = st.text_input("Ask about books here:", key="chat")
        if chat_input:
            response = query_model(chat_input)
            st.write(response)

if __name__ == "__main__":
    main()
