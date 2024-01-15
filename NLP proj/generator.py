'''
检索相关信息, 根据不同的prompt生成文档
'''
import openai
from openai import OpenAI
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np 


client = OpenAI(
    # 输入转发API Key
    api_key="sk-xxx",
    base_url="https://api.chatanywhere.com.cn/v1"
)


'''
# 调用gpt进行信息检索
def retrieve1(key_word):
    prompt = f"Assuming you are an information retriever, you have no prior knowledge and can only retrieve from the materials I provide. 
            Now I need you to retrieve the 20 most relevant sentences related to {key_word} from the reference materials I provide.
            The reference materials are as follows:"
    prompt += texts
    messages = [
        {'role':'user', 'content':prompt}
    ]
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4", messages=messages
        )
        reply = chat_completion.choices[0].message.content
    except Exception as e:
        reply = None
        print(f"Error, {e}")

    return reply
'''

# 检索与主题相关的句子
def retrieve(texts, keywords):
    sentences = sent_tokenize(texts)

    vectorizer = TfidfVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(sentences)
    keyword_vectors = vectorizer.transform(keywords)

    similarities = cosine_similarity(keyword_vectors, sentence_vectors)

    average_similarities = np.mean(similarities, axis = 0)

    top_idx = np.argsort(average_similarities)[-100:]

    top_sentences = [sentences[i] for i in top_idx]

    return top_sentences

#文档生成
def generate(start_prompt, total_prompt, end_prompt, method):
    prompt = start_prompt + total_prompt[method] + end_prompt
    messages = [
        {'role':'user', 'content':prompt}
    ]
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-4-1106-preview", messages=messages
        )
        reply = chat_completion.choices[0].message.content
    except Exception as e:
        reply = None
        print(f"Error, {e}")

    return reply


def main():
    with open("wiki_cleaned.txt", 'r', encoding='utf-8') as f:
        texts = f.read()

    with open('prompt.json', 'r') as f:
        total_prompt = json.load(f)

    key_word = ""

    shared_prompt = f"You are a Encyclopedia document generator, you have no prior knowledge, and can only generate text with a structured format based on the information I provide, without including extra sentences. The key word is:{key_word}."
    end_prompt = 'Now please generate a Encyclopedia document. Make sure the information only comes from the references provided. Do not add any additional information. References are as follows:' + texts

    # CoT
    generation_CoT = generate(shared_prompt, total_prompt, end_prompt, 'CoT')
    with open('generation_CoT.txt', 'w', encoding='utf-8') as f:
        f.write(generation_CoT)

    # ICL
    generation_ICL = generate(shared_prompt, total_prompt, end_prompt, 'ICL')
    with open('generation_ICL.txt', 'w', encoding='utf-8') as f:
        f.write(generation_ICL)

    # template
    keywords = ['education', 'school', 'degree', 'student', 'work', 'profession', 'company', 'internship', 'experience', 'family', 'home', 'member', 'skill', 'ability', 'competence', 'hometown', 'birth', 'city', 'country', 'hobby', 'life', 'achievement', 'awards', 'value', 'principles', 'culture', 'marriage', 'parenthood']
    ref_list = retrieve(texts, keywords)
    end_prompt_ = ''
    for ref in ref_list:
        end_prompt_ += ref + '\n'
    generation_template = generate(shared_prompt, total_prompt, end_prompt_, 'template')
    with open('generation_template.txt', 'w', encoding='utf-8') as f:
        f.write(generation_template)


if __name__ == "__main__":
    main()


