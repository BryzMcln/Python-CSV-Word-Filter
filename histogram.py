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
    "can",
    "will",
    "just",
    "don",
    "should",
    "may",
    "s",
    "now",
    "receive",
    "received",
    "well",
    "much",
    "buy",
    # tagalog
    "bili",
    "na",
    "ng",
    "nang",
    "sa",
    "pa",
    "ba",
    "po",
    "mag",
    "nag",
    "pag",
    "lang",
    "ang",
    "ako",
    "ko",
    "yung",
    "ung",
    "naman",
    "sya",
    "nya",
    "jan",
    "nyan",
    "din",
    "pala",
    "sana",
    "pero",
    "hindi",
    "di",
    "oo",
    "siguro",
    "pwede",
    "parang",
    "dumating",
    "talaga",
    "order",  # order
    "quality",  # quality
    "product",  # product
    "item",  # item
    "appearance",  # appearance
    "feature",  # feature
    "seller",  # seller
    "shop",  # shop
    "price",  # price
    "colour",
    "color",  # color
    "delivery",  # delivery
    "money",  # money
    "value",  # value
    "material",  # material
    "performance",  # performance
    "rider",  # rider
    "box",  # box
    "controller",  # controller
    "shopee",
]


# Mapping for variations to combine
same_value = {
    "thank": "thanks",
    "ok": "okay",
}


def txt_process(comment):
    words = re.findall(r"\b[^\d\W]+\b", comment.lower())
    # Apply variation mapping
    words = [same_value.get(word, word) for word in words]
    return [w for w in words if w not in stop_words]


def histogram(dataset, word_count=15):
    all_words = []
    for index, data in dataset.iterrows():
        words = txt_process(str(data["comment"]))
        all_words.extend(words)

    word_freq = FreqDist(all_words)
    top_words = word_freq.most_common(word_count)
    top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(15, 9))
    sns.barplot(x="Frequency", y="Word", data=top_words_df)
    plt.title("Most Common Words in Gaming Product Comments")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("data.csv", encoding="utf-8")

    # Plot word histogram
    histogram(dataset, word_count=15)
