import pandas as pd
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
import seaborn as sns


def read_data(file_path):
    try:
        dataset = pd.read_csv(file_path, encoding="utf-8")
        return dataset
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


def process_data(dataset, target_words):
    dataset["combined_text"] = (
        dataset[["comment", "rating_star", "itemid"]].astype(str).agg(" ".join, axis=1)
    )

    word_freq = {word: 0 for word in target_words}

    for index, data in dataset.iterrows():
        comment = str(data["combined_text"]).lower()
        for word in target_words:
            if word in comment:
                word_freq[word] += 1

    return pd.DataFrame(list(word_freq.items()), columns=["Word", "Frequency"])


def plot_bar_chart(new_dataframe):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=new_dataframe["Frequency"], y=new_dataframe["Word"])
    plt.title("Word Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tick_params(axis="both", which="major", labelsize=12)  # Adjust font size
    plt.tight_layout()  # Adjust layout
    plt.show()


if __name__ == "__main__":
    file_path = "data.csv"
    target_words = [
        "thanks",
        "thank",
        "good",
        "ok",
        "okay",
        "nice",
        "cheap",
        "satisfied",
        "simple",
        "well",
        "easy",
        "work",
        "better",
        "best",
        "great",
        "fast",
        "superb",
        "quick",
        "cool",
        "love",
        "like",
        "pretty",
        "cute",
        "excellent",
        "very",
        "worth",
        "super",
        "perfect",
        "sakto",
        "sulit",
        "medyo",
        "maganda",
    ]

    dataset = read_data(file_path)

    if dataset is not None:
        new_dataframe = process_data(dataset, target_words)
        plot_bar_chart(new_dataframe)
