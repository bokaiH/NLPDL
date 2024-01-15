'''
利用爬虫收集参考资料并去噪
'''

import requests
import re
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

# 在维基百科上搜索关键字，返回第一个结果的标题
def search_wiki(keyword):
    url =  f"https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": keyword
    }

    response = requests.get(url=url, params=params)
    results = response.json()

    if results.get("query") is None:
        return "No results found."
    else:
        data = results["query"]["search"]
        return data[0]["title"]

# 获取维基百科页面的参考资料
def get_references(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    references = soup.find(id="References")
    if not references:
        return []

    refs_list = references.find_next("ol")
    links = refs_list.find_all("a", href=True)
    return [link['href'] for link in links if link['href'].startswith('http')]

# 获取参考资料的内容
def get_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    content = soup.find('article') or soup.find('div', class_='main-content')
    if not content:
        return ''
    
    text = ''
    for elem in content.find_all(['p', 'h1', 'h2', 'h3', 'li']):  # 包含更多可能包含文本的标签
        elem_text = elem.get_text().strip()
        if not re.search(r'Copyright|All rights reserved|By\s+\w+', elem_text):
            text += elem_text + '\n'
    return text

# 清洗文本
def clean_text(text):
    sentences = text.split('\n')
    result = ''
    unique_sentences = set()

    for sentence in sentences:
        if sentence.strip() == '':
            continue

        if len(sentence.split()) < 5:
            continue

        if re.search(r'\b(Cookie|Privacy policy|Terms of use|Report|Contact)\b', sentence, flags=re.IGNORECASE):
            continue

        sentence = re.sub(r'http\S+', '', sentence)
        sentence = re.sub(r'\[.*?\]', '', sentence)
        
        if sentence.endswith('.') or sentence.endswith('?') or sentence.endswith('!') or sentence.endswith('"') or sentence.endswith(","):
            sentence = re.sub(r'[^a-zA-Z0-9.,;:!?\'\"-]', '', text)
            if sentence not in unique_sentences:
                unique_sentences.add(sentence)
                result += sentence + '\n'

    return result


def main():
    keyword = input("Enter a keyword: ")
    title = search_wiki(keyword)
    references = get_references(title)
    for link in references[:30]:
        text = get_content(link)
        if len(text.split('\n')) < 3:
            continue
        with open('wiki_cleaned.txt','a',encoding='utf-8') as f:
            f.write(clean_text(text))
        with open('wiki_raw.txt','a',encoding='utf-8') as f:
            f.write(text)
    return


if __name__ == "__main__":
    main()