# **Overview**

This project assembles a set of experiments that compare how the concepts `nature`, `resilience`, `sustainability`, and `climate change` relate to each other across multiple text sources. The notebook `main2.ipynb` guides you through:

- Collecting Wikipedia pages and exporting their raw content.
- Building TF-IDF corpora from full articles, subsection titles, and linked entities.
- Measuring concept similarity with cosine distance, visual heatmaps, and tabular reports.
- Comparing lexical resources (WordNet) with distributional methods (Word2Vec embeddings).
- Expanding the entity graph by scraping the top linked Wikipedia pages.
- Pulling recent news via NewsAPI and evaluating cross-keyword similarity in journalism.

Most steps persist intermediate artefacts (CSV tables, JSON exports, PNG plots, pickled datasets) inside the project root so results can be reused across later analyses.

# **Repository Contents**

- `main2.ipynb` – primary workflow covering all research tasks.
- `task9_news.py` – standalone helper used by Task 9 (news ingestion and similarity analysis).
- `wikipedia_*.json`, `wikipedia_all_keywords.json` – cached Wikipedia responses saved by Task 1.
- `similarity_*.csv`, `similarity_table.txt`, `_heatmap.png`, `wordnet_similarity_matrices.png` – generated outputs from similarity experiments.
- `expanded_entity_data.pkl`, `expanded_corpus.json`, `expanded_entity_similarities.png` – entity expansion cache and reports.

# **Prerequisites**

- Python 3.9 or newer.
- Internet access (Wikipedia API, NewsAPI, gensim model downloads).
- A NewsAPI key stored in a local `.env` file (see **Environment Variables**). For now, there have been attached the env file with API KEY

Recommended Python packages (install with `pip install ...`):

```
pip install wikipedia-api nltk scikit-learn pandas numpy matplotlib seaborn requests beautifulsoup4 python-dotenv wordcloud gensim jupyter

```

# **NLTK Resources**

Several preprocessing steps depend on NLTK tokenizers, stop words, and WordNet data. Run once before executing the notebook:

```
python -m nltk.downloader punkt stopwords wordnet omw-1.4

```

# **Environment Variables**

Create a `.env` file in the project root containing your NewsAPI credentials: (For now, there is the env file with my api key)

```
NEWSAPI_KEY=your_newsapi_key_here

```

# **Running the Notebook**

1. Launch Jupyter Lab or the classic notebook interface.
2. Open `main2.ipynb` and execute cells sequentially. The notebook is organised in logical sections; you can run only the portions you need:
    - **Task 1 – Wikipedia Harvest**: Downloads article bodies, subsection titles, and entity links for the four keywords and writes JSON caches.
    - **Task 2 – Corpus Similarity**: Preprocesses article text, builds TF-IDF vectors, prints similarity matrices, and generates `similarity_matrix.csv` plus `similarity_heatmap.png`.
    - **Task 3 & 4 – Subsections vs Entities**: Repeats the similarity workflow using subsection titles or linked entity names as documents, saving CSVs and `comparison_heatmaps.png`.
    - **Task 5 – WordNet Baseline**: Uses Wu-Palmer, path, and Leacock-Chodorow similarities to compare keywords from a lexical perspective; exports `similarity_matrix_wordnet_wup.csv` and `wordnet_similarity_matrices.png`.
    - **Task 6 – Entity Expansion**: Scrapes the first 30 linked entities per keyword, collects categories and summaries, and saves the expanded corpora (`expanded_entity_data.pkl`, `expanded_corpus.json`). Scraping respects a 0.5 s delay between requests; adjust `max_entities_per_keyword` or `delay` if needed.
    - **Task 7 – Expanded Entity Similarity**: Evaluates the expanded representations and stores heatmaps plus CSV matrices for categories, summaries, and linked-entity signals.
    - **Task 8 – Word2Vec Embeddings**: Downloads `glove-wiki-gigaword-100` via `gensim`, derives embeddings for each keyword, and outputs `similarity_matrix_word2vec.csv` along with a PCA scatter plot.
    - **Task 9 – News Similarity**: Delegates to `task9_news.py` to fetch recent news articles, build word clouds, and compute TF-IDF and Word2Vec similarities across rolling monthly windows.

## **Running Task 9 from the Command Line**

If you prefer to execute the news ingestion outside the notebook:

```
python task9_news.py

```

The script downloads the `word2vec-google-news-300` model (~1.6 GB) on first run, so ensure adequate disk space and patience. For each (start, end) monthly window it saves datasets (`news_dataset_*.csv`) and similarity matrices (`tfidf_similarity_*.csv`, `w2v_similarity_*.csv`).

**Reusing Saved Artefacts**

- The notebook reads cached JSON/CSV files if they already exist, allowing you to skip expensive scraping or API calls after the first run.
- Delete or rename generated files if you want to rerun a section from scratch.

## **Troubleshooting**

- **HTTP 429 (Too Many Requests)**: Increase the `delay` parameter in `WikipediaEntityExpander` or limit `max_entities_per_keyword`.
- **Missing NLTK resource**: Re-run `python -m nltk.downloader ...` for the listed corpora.
- **NewsAPI errors**: Verify your key in `.env` and ensure the account has sufficient quota.
- **gensim download failures**: Check network connectivity or switch to a smaller model (e.g., `glove-wiki-gigaword-50`).

### **Next Steps**

- Extend `keywords` to analyse additional environmental concepts.
- Incorporate sentiment analysis or topic modelling on the news corpus.
- Automate the workflow by converting notebook sections into standalone Python scripts.
