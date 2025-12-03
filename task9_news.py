import os
from dotenv import load_dotenv
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import time
import gensim.downloader as api
from task2_tfidf import preprocess

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    raise ValueError("Please set your NEWSAPI_KEY in the .env file.")

print("Loading Word2Vec model (this may take a while)...")
w2v_model = api.load("word2vec-google-news-300")  

keywords = ["nature", "resilience", "sustainability", "climate change"]

def get_time_periods(months=1):
    """Return list of (start_date, end_date) strings for each month period."""
    end_date = datetime.now()
    periods = []
    for i in range(months):
        start = (end_date - timedelta(days=30*(i+1))).strftime("%Y-%m-%d")
        end = (end_date - timedelta(days=30*i)).strftime("%Y-%m-%d")
        periods.append((start, end))
    return periods[::-1] 

def fetch_news(keywords, from_to_period):
    """Fetch news for given keywords and period. Returns DataFrame."""
    all_articles = []
    from_param, to_param = from_to_period

    for kw in keywords:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={kw}&from={from_param}&to={to_param}&language=en&"
            f"sortBy=relevancy&pageSize=100&apiKey={API_KEY}"
        )
        r = requests.get(url)
        data = r.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            for article in articles:
                all_articles.append({
                    "keyword": kw,
                    "from_date": from_param,
                    "to_date": to_param,
                    "date": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", "")
                })
        else:
            print(f"Error fetching '{kw}' ({from_param} to {to_param}): {data.get('message')}")
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    return df

def generate_wordcloud(df, period_label):
    """Generate WordClouds for each keyword in a DataFrame."""
    if df.empty:
        print(f"No news found for period {period_label}, skipping WordCloud.")
        return

    for kw in df["keyword"].unique():
        subset = df[df["keyword"] == kw]
        if subset.empty:
            continue
        text = " ".join(subset["content"].dropna().tolist())
        if not text.strip():
            continue
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud for '{kw}' ({period_label})")
        plt.show()

def compute_tfidf_similarity(df):
    """Compute similarity between keywords using TF-IDF and cosine similarity."""

    if df.empty:
        print("No data available!")
        return pd.DataFrame()

    grouped_contents = {}
    for keyword in df['keyword'].unique():
        articles = df[df['keyword'] == keyword]['content']
        combined_text = " ".join(articles.dropna())
        grouped_contents[keyword] = combined_text
        print(f"Keyword '{keyword}': {len(combined_text.split())} words combined")

    preprocessed_contents = {}
    for keyword, text in grouped_contents.items():
        preprocessed_contents[keyword] = preprocess(text)

    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_contents.values())

    cos_sim = cosine_similarity(tfidf_matrix)

    sim_df = pd.DataFrame(cos_sim, index=preprocessed_contents.keys(), columns=preprocessed_contents.keys())

    return sim_df

def compute_w2v_similarity(df):
    """Compute similarity between keywords using Word2Vec embeddings."""

    if df.empty:
        print("No data available!")
        return pd.DataFrame()

    grouped_contents = {}
    for keyword in df['keyword'].unique():
        articles = df[df['keyword'] == keyword]['content']
        combined_text = " ".join(articles.dropna())
        grouped_contents[keyword] = combined_text
        print(f"Keyword '{keyword}': {len(combined_text.split())} words combined")

    preprocessed_contents = {}
    for keyword, text in grouped_contents.items():
        preprocessed_contents[keyword] = preprocess(text)

    keywords_list = list(preprocessed_contents.keys())
    sim_matrix = pd.DataFrame(index=keywords_list, columns=keywords_list, dtype=float)

    for i, kw1 in enumerate(keywords_list):
        words1 = [w for w in preprocessed_contents[kw1].split() if w in w2v_model.key_to_index]

        for j, kw2 in enumerate(keywords_list):
            words2 = [w for w in preprocessed_contents[kw2].split() if w in w2v_model.key_to_index]

            if words1 and words2:
                sim = w2v_model.n_similarity(words1, words2)
            else:
                sim = 0

            sim_matrix.loc[kw1, kw2] = sim

    return sim_matrix

commentry =  commentary = """
    # Step 11: Commentary on Similarity Results

    The analysis of news content across the keywords **nature**, **resilience**, **sustainability**, and **climate change** shows two types of similarity: lexical similarity (TF-IDF) and semantic similarity (Word2Vec).

    ## Observations

    ### TF-IDF Similarity
    - Values range between 0.68 – 0.73 for most keyword pairs.
    - Indicates moderate lexical overlap in the news articles; keywords like nature and sustainability may share common vocabulary, but wording differs across articles.
    - TF-IDF captures frequency of exact terms but cannot detect synonymy or contextual similarity.

    ### Word2Vec Similarity
    - Values are very high, between 0.94 – 0.97, indicating strong semantic similarity.
    - Even when exact words differ, Word2Vec embeddings detect that these concepts occur in similar contexts across the news articles.
    - This aligns with the expectation that news about climate change often involves discussions on resilience and sustainability, even if different terms are used.

    ## Interpretation
    - TF-IDF results reflect lexical overlap, capturing direct word usage. Moderate values suggest that while keywords are related, journalists use varied vocabulary.
    - Word2Vec results reflect semantic similarity, capturing underlying meaning. High similarity indicates that these concepts are conceptually connected in the news discourse.
    - Comparing TF-IDF and Word2Vec highlights the difference between surface-level word matching versus deep contextual understanding.

    ## Implications
    - For trend analysis and topic clustering in news, Word2Vec or other embedding-based approaches may give a more meaningful similarity measure.
    - TF-IDF is still useful for understanding word usage patterns and keyword co-occurrences.

    ## Literature References
    - Manning et al., 2008 – *Introduction to Information Retrieval*: TF-IDF captures term frequency importance but is limited to lexical similarity.
    - Mikolov et al., 2013 – *Efficient Estimation of Word Representations in Vector Space*: Word2Vec embeddings capture semantic similarity from context co-occurrence.
    """
if __name__ == "__main__":
    periods = get_time_periods(months=1) 

    all_data = []
    for start, end in periods:
        period_label = f"{start}_to_{end}"
        print(f"Fetching news for period: {period_label}")

        df_period = fetch_news(keywords, (start, end))
        if df_period.empty:
            print(f"No news found for period {period_label}, skipping.")
            continue
        print(period_label, 'PERIOD LABEL')

        df_period.to_csv(f"n1ews_dataset_{period_label}.csv", index=False)
        generate_wordcloud(df_period, period_label)

        tfidf_sim = compute_tfidf_similarity(df_period)
        tfidf_sim.to_csv(f"tfidf_similarity_{period_label}.csv")

        w2v_sim = compute_w2v_similarity(df_period)
        w2v_sim.to_csv(f"w2v_similarity_{period_label}.csv")

        all_data.append(df_period)

    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv("news_dataset_all_periods.csv", index=False)
    


