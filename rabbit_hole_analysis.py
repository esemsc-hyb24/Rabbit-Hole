import os
import pandas as pd
import numpy as np
import torch
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# === Load precomputed embeddings and base dataframe ===
df = pd.read_csv("embedded_df.csv")
embedding_matrix = torch.load("embeddings.pt")
embedding_array = embedding_matrix.numpy()
df["embedding"] = list(embedding_array)

# === Step 1: Dimensionality Reduction ===
umap_reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=15,
    metric='cosine',
    random_state=42
)
embedding_15d = umap_reducer.fit_transform(embedding_array)

# === Step 2: HDBSCAN Clustering ===
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    metric='euclidean',
    prediction_data=True,
    gen_min_span_tree=True
)
cluster_labels = clusterer.fit_predict(embedding_15d)
df["cluster"] = cluster_labels

n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = np.sum(cluster_labels == -1)

# === Step 3: Extract Top Keywords Per Cluster ===
def extract_top_keywords_c_tf_idf(df, text_col='text', label_col='cluster', top_k=20):
    cluster_texts = (
        df[df[label_col] != -1]
        .groupby(label_col)[text_col]
        .apply(lambda texts: " ".join(texts))
    )

    count_vec = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
    count_matrix = count_vec.fit_transform(cluster_texts)
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
    ctfidf_matrix = tfidf.fit_transform(count_matrix)

    feature_names = count_vec.get_feature_names_out()
    records = []
    for idx, row in enumerate(ctfidf_matrix):
        cluster_id = cluster_texts.index[idx]
        sorted_indices = row.toarray().flatten().argsort()[::-1][:top_k]
        keywords = [feature_names[i] for i in sorted_indices]
        count = df[df[label_col] == cluster_id].shape[0]
        records.append({
            "cluster": cluster_id,
            "keywords": ", ".join(keywords),
            "count": count
        })

    return pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)

cluster_summary = extract_top_keywords_c_tf_idf(df)

# === Step 4: Label Clusters Using Ollama ===
def label_clusters_with_ollama_df(summary_df, model_name="gemma3:4b"):
    llm = Ollama(model=model_name)
    prompt_template = PromptTemplate.from_template(
        "You are helping categorize clusters of search history topics.\n"
        "Given the following top keywords, generate a short and specific label "
        "that summarizes the main idea of the cluster using 2–4 words. Avoid vague or generic terms.\n\n"
        "Keywords: {keywords}\nLabel:"
    )
    labels = []
    for _, row in summary_df.iterrows():
        prompt = prompt_template.format(keywords=row["keywords"])
        try:
            label = llm.invoke(prompt).strip()
        except Exception as e:
            print(f"❌ Error for cluster {row['cluster']}: {e}")
            label = "Unknown"
        labels.append(label)

    summary_df = summary_df.copy()
    summary_df["label"] = labels
    return summary_df

cluster_summary = label_clusters_with_ollama_df(cluster_summary)
label_map = dict(zip(cluster_summary["cluster"], cluster_summary["label"]))
df["cluster_label"] = df["cluster"].apply(lambda cid: label_map.get(cid, "Noise") if cid != -1 else "Noise")

# === Step 5: UMAP 2D for Plotting ===
umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
embedding_2d = umap_2d.fit_transform(embedding_array)
df["x_2d"] = embedding_2d[:, 0]
df["y_2d"] = embedding_2d[:, 1]

# === Step 6: Save UMAP Cluster Plot ===
plt.figure(figsize=(10, 7))
unique_clusters = sorted(df["cluster"].unique())
palette = sns.color_palette("husl", len(unique_clusters))
cluster_color_map = {cid: palette[i] for i, cid in enumerate(unique_clusters)}

for cid in unique_clusters:
    subset = df[df["cluster"] == cid]
    label = label_map.get(cid, "Noise") if cid != -1 else "Noise"
    plt.scatter(subset["x_2d"], subset["y_2d"], s=60, label=label, color=cluster_color_map[cid], alpha=0.7)

plt.title("Search History Clusters (UMAP 2D)")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.legend(title="Cluster Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("clusters.png")
plt.close()

# === Step 7: Sentiment Analysis ===
analyser = SentimentIntensityAnalyzer()

def extract_clusters_from_df(df):
    return [
        {
            "topic": row["label"],
            "keywords": [kw.strip().lower() for kw in row["keywords"].split(",")],
            "num_points": row["count"]
        }
        for _, row in df.iterrows()
    ]

def classify_word_sentiment(word):
    score = analyser.polarity_scores(word)['compound']
    return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"

def classify_all_keywords(keywords):
    return [classify_word_sentiment(word) for word in keywords]

def plot_keyword_sentiment_pie(sentiments):
    counts = Counter(sentiments)
    labels, sizes = counts.keys(), counts.values()
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["green", "red", "grey"])
    plt.title("Sentiment Breakdown of Keywords")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("sentiment_pie.png")
    plt.close()

def plot_topic_names(clusters):
    topic_counts = {c["topic"]: c["num_points"] for c in clusters}
    sorted_topics = dict(sorted(topic_counts.items(), key=lambda item: item[1], reverse=True))
    plt.figure(figsize=(9, 9))
    plt.pie(sorted_topics.values(), labels=sorted_topics.keys(), autopct='%1.1f%%', startangle=90)
    plt.title("Cluster Topic Distribution")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("topics_pie.png")
    plt.close()

def detect_sentiment_bubble(sentiments, threshold=0.7):
    counts = Counter(sentiments)
    total = len(sentiments)
    for sentiment, count in counts.items():
        if count / total > threshold:
            return f"Sentiment Bubble Detected: '{sentiment.upper()}' dominates {int(count / total * 100)}% of all keywords."
    return None

def detect_sentiment_warnings(sentiments, threshold=0.6):
    warnings = []
    counts = Counter(sentiments)
    total = len(sentiments)
    for sentiment, count in counts.items():
        pct = count / total
        if pct > threshold:
            msg = f"{sentiment.upper()} sentiment dominates {int(pct * 100)}% of keywords."
            warnings.append(msg)
    return warnings

clusters = extract_clusters_from_df(cluster_summary)
all_keywords = [kw for cluster in clusters for kw in cluster["keywords"]]
keyword_sentiments = classify_all_keywords(all_keywords)

plot_topic_names(clusters)
plot_keyword_sentiment_pie(keyword_sentiments)
bubble_warning = detect_sentiment_bubble(keyword_sentiments)
sentiment_warnings = detect_sentiment_warnings(keyword_sentiments)

# === Step 8: Save Analysis Report ===
with open("analysis_summary.txt", "w") as f:
    f.write(f"Total Clusters: {n_clusters}\n")
    f.write(f"Noise Points: {n_noise}\n\n")
    f.write("Cluster Summary:\n")
    for _, row in cluster_summary.iterrows():
        f.write(f"Cluster {row['cluster']}: {row['label']} ({row['count']} items)\n")
        f.write(f"  Keywords: {row['keywords']}\n\n")

    f.write("Sentiment Summary:\n")
    for sentiment, count in Counter(keyword_sentiments).items():
        f.write(f"{sentiment.capitalize()}: {count} keywords\n")

    if bubble_warning:
        f.write(f"\n⚠️ {bubble_warning}\n")
    if sentiment_warnings:
        f.write("\n".join(["⚠️ " + w for w in sentiment_warnings]))