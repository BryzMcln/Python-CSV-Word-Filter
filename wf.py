import pandas as pd
import re
from nltk.probability import FreqDist
from matplotlib import pyplot as plt
import seaborn as sns

# Specify the words you want to track
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

word_count = len(target_words)
dataset = pd.read_csv("data.csv", encoding="utf-8")
print(dataset)

# Combine 'comment', 'rating_star', and 'itemid' columns into one string
dataset["combined_text"] = (
    dataset[["comment", "rating_star", "itemid"]].astype(str).agg(" ".join, axis=1)
)

# Initialize a dictionary to store word frequencies
word_freq = {word: 0 for word in target_words}

for index, data in dataset.iterrows():
    comment = str(
        data["combined_text"]
    ).lower()  # Updated to 'combined_text' column and converted to lowercase

    # Iterate through target words and count occurrences
    for word in target_words:
        if word in comment:
            word_freq[word] += 1

# Create a DataFrame from the word frequencies
new_dataframe = pd.DataFrame(list(word_freq.items()), columns=["Word", "Frequency"])

# Sort the DataFrame by frequency in descending order
new_dataframe = new_dataframe.sort_values(by="Frequency", ascending=False)

# Plot the bar chart in fullscreen mode without toolbar
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.barplot(x=new_dataframe["Frequency"], y=new_dataframe["Word"])
plt.title("Word Frequencies")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show(block=False)
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()
