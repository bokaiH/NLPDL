'''
去噪效果评估:单词数量,句子数量,唯一单词数量
'''
from collections import Counter
import matplotlib.pyplot as plt


def text_statistics(text):
    word_count = len(text.split())
    sentence_count = text.count('.')
    return word_count, sentence_count

with open('wiki_raw.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
with open('wiki_cleaned.txt', 'r', encoding='utf-8') as f:
    cleaned_text = f.read()
    
raw_word_count, raw_sentence_count = text_statistics(raw_text)
cleaned_word_count, cleaned_sentence_count = text_statistics(cleaned_text)

raw_unique_words = len(Counter(raw_text.split()))
cleaned_unique_words = len(Counter(cleaned_text.split()))

categories = ['Word Count', 'Sentence Count', 'Unique Words']
raw_values = [raw_word_count, raw_sentence_count, raw_unique_words]
cleaned_values = [cleaned_word_count, cleaned_sentence_count, cleaned_unique_words]

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

bar1 = plt.bar([x - bar_width/2 for x in range(len(categories))], raw_values, bar_width,
alpha=opacity, color='b', label='Raw')

bar2 = plt.bar([x + bar_width/2 for x in range(len(categories))], cleaned_values, bar_width,
alpha=opacity, color='g', label='Cleaned')

plt.xlabel('Metrics')
plt.ylabel('Counts')
plt.title('Text De-noising Analysis')
plt.xticks(range(len(categories)), categories)
plt.legend()

plt.tight_layout()
plt.show()
