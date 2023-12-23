import pandas as pd
import re
import nltk
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords  # words in english


# List of common English stop words
stop_words = set(stopwords.words("english"))

# Add Tagalog stop words
tagalog_stop_words = [
    "bili","na","ng","nang","sa","pa","ba","po","mag","nag","pag","lang","ang","ako","ko","yung","ung","naman",
    "sya","siya","niya","nya","jan","nyan","din","rin","lahat","kaya","ito","pala","sana","pero","hindi","ulit",
    "mga","di","oo","siguro","kasi","dahil","pwede","parang","dumating","lng","para","talaga",
    "may", "also", "got",
]

unecessary_words = [
    "quality", "shopee", "order", "seller", "shop",   "colour", "color",
    "delivery", "shipping",  "money",  "material",   "rider", "box", "packaging", "packed",
]
stop_words.update(tagalog_stop_words, unecessary_words)

# Mapping for variations to combine
same_value = {
    "thank": "thanks",
    "good": "goods",
    "ok": "okay",
    "maganda": "ganda",
    "working": "works",
    "items": "item",
}


def processing_each_and_every_text(comment):
    # Exclude all single letters like a, b, c ...
    words = re.findall(r"\b[^\d\W_]+\b", comment.lower())
    return [
        # count the same value
        same_value.get(w, w)
        for w in words
        if w not in stop_words and len(w) > 1
    ]


def histogram(dataset, word_count=15):
    all_words = []
    for index, data in dataset.iterrows():
        words = processing_each_and_every_text(str(data["comment"]))
        all_words.extend(words)

    word_freq = FreqDist(all_words)
    top_words = word_freq.most_common(word_count)
    top_words_df = pd.DataFrame(top_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(15, 9))
    sns.barplot(x="Frequency", y="Word", data=top_words_df)
    plt.title("Most Common Words in Gaming Product Comments")
    plt.xlabel("Frequency")
    plt.ylabel("Words Used")
    plt.show()


if __name__ == "__main__":
    # Load csv file
    dataset = pd.read_csv("data.csv", encoding="utf-8")

    # histogram parameter
    histogram(dataset, word_count=20)  # top 20 words
