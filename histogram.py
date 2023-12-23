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
    "get",
    "got",
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
    "really",
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
    "now",
    "receive",
    "received",
    "well",
    "much",
    "buy",
    "also",
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
    "siya",
    "niya",
    "nya",
    "jan",
    "nyan",
    "din",
    "rin",
    "lahat",
    "kaya",
    "ito",
    "pala",
    "sana",
    "pero",
    "hindi",
    "ulit",
    "mga",
    "di",
    "oo",
    "siguro",
    "kasi",
    "dahil",
    "pwede",
    "parang",
    "dumating",
    "lng",
    "para",
    "talaga",
    "order",  # order
    "quality",  # quality
    "product",  # product
    "items",
    "item",  # item
    "appearance",  # appearance
    "feature",  # feature
    "seller",  # seller
    "shop",  # shop
    "price",  # price
    "colour",
    "color",  # color
    "delivery",  # delivery
    "shipping",  # shipping
    "money",  # money
    "value",  # value
    "material",  # material
    "performance",  # performance
    "rider",  # rider
    "box",  # box
    "packaging",
    "packed",
    "controller",  # controller
    "keyboard",  # keyboard
    "mouse",  # mouse
    "time",  # time
    "shopee",
]

# combine count same word meaning
same_value = {
    "thank": "thanks",
    "good": "goods",
    "ok": "okay",
    "maganda": "ganda",
    "working": "works",
}


def proecessing_each_and_every_txt(comment):
    words = re.findall(r"\b[^\d\W_]+\b", comment.lower())  # Exclude all single letters
    return [
        same_value.get(w, w) for w in words if w not in stop_words and len(w) > 1
    ]  # count the same value


def histogram(dataset, word_count=15):
    all_words = []
    for index, data in dataset.iterrows():
        words = proecessing_each_and_every_txt(str(data["comment"]))
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

    # histogram parameter
    histogram(dataset, word_count=20)
