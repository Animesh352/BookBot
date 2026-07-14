# BookBot

A Streamlit book recommendation and Q&A chatbot. Search for a book by title or description, get similar recommendations retrieved from a Pinecone vector index, and chat with the bot about any book using GPT-3.5-turbo.

## How it works

1. Search query is embedded with `sentence-transformers/all-MiniLM-L6-v2` (runs locally via the `transformers` library).
2. The embedding is sent to a Pinecone index that stores metadata for a large book catalogue (title, author, year, publisher, language, category, summary).
3. Top-1 nearest neighbor returns the matched book's metadata; top-10 returns similar books.
4. Summaries that are short or non-English are expanded into 150-300 word English paragraphs via GPT-3.5-turbo.
5. A persistent chat session backed by GPT-3.5-turbo lets you ask questions about any book.

## Built with

- **Streamlit** -- UI and session state management
- **Pinecone** -- vector index for book metadata and similarity search
- **OpenAI GPT-3.5-turbo** -- summary expansion and conversational Q&A
- **sentence-transformers/all-MiniLM-L6-v2** -- local text embedding model (via HuggingFace `transformers`)

## UI

Two tabs:

| Tab | What it shows |
|-----|---------------|
| Recommended Books | Top-10 most similar books to the search query, each with cover image, metadata, and an expanded English summary |
| Chat | Persistent multi-turn chat with BookBot; context resets when a new book is searched |

The sidebar displays the matched book's cover image, title, author, year, publisher, language, category, and an expanded summary.

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key (with a pre-populated index -- see Data below)

### Install

```bash
git clone https://github.com/Animesh352/BookBot.git
cd BookBot
pip install -r requirements.txt
```

### Set environment variables

macOS / Linux:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
export PINECONE_API_KEY=your_pinecone_api_key_here
```

Windows:

```bash
set OPENAI_API_KEY=your_openai_api_key_here
set PINECONE_API_KEY=your_pinecone_api_key_here
```

### Run

```bash
streamlit run app.py
```

## Data

Book metadata is stored in chunked CSV files under `Data/` (`df_chunk_1.csv` through `df_chunk_14.csv`). Each row contains at minimum: `book_title`, `book_author`, `year_of_publication`, `publisher`, `Language`, `Category`, `Summary`, `img_l` (cover image URL).

You need to load this data into a Pinecone index with embeddings generated from `utils.py`'s `text_to_embedding()` before the app can run. The embedding model is `sentence-transformers/all-MiniLM-L6-v2`.

## Language support

The app recognizes and displays language names for 30+ languages including Arabic, Chinese (Simplified/Traditional), French, German, Hindi, Japanese, Korean, Russian, Spanish, and more.

## Author

Animesh Giri -- [Animesh352](https://github.com/Animesh352)
