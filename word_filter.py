import pandas as pd
import re
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
import seaborn as sns

# List of common English stop words
stop_words = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ma",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
]


def preprocess_text(comment):
    words = re.findall(r"\b\w+\b", comment.lower())
    return [w for w in words if w not in stop_words]


def plot_word_histogram(dataset, word_count=15):
    all_words = []
    for index, data in dataset.iterrows():
        words = preprocess_text(str(data["comment"]))
        all_words.extend(words)

    word_freq = FreqDist(all_words)
    top_words = word_freq.most_common(word_count)
    top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(12, 8))
    sns.barplot(x="Frequency", y="Word", data=top_words_df)
    plt.title("Most Common Words in Gaming Product Comments")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("data.csv", encoding="utf-8")

    # Plot word histogram
    plot_word_histogram(dataset, word_count=15)
