'''
评估生成文档质量
'''
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score


#事实性评估
nlp = spacy.load("en_core_web_sm")

# 抽取命名实体
def extract_entities(doc):
    spacy_doc = nlp(doc)
    entities = [(ent.text, ent.label_) for ent in spacy_doc.ents]
    return entities


def fact_check(generated_doc, reference_doc):
    gen_ent_set = set(generated_doc)
    ref_ent_set = set(reference_doc)
    common_entities = gen_ent_set.intersection(ref_ent_set)
    missing_or_incorrcet_entities = gen_ent_set.difference(ref_ent_set)

    return {
        "common_entities": list(common_entities),
        "missing_or_incorrcet_entities": list(missing_or_incorrcet_entities)
    }


#信息性评估
def get_similarity_score(doc1, doc2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix)[0][1]


# 篇章结构评估
# 分析文档结构
def analyze_structure(doc):
    spacy_doc = nlp(doc)
    paragraphs = [p.text for p in spacy_doc.sents if '\n' in p.text]
    titles = [p.text for p in spacy_doc.sents if ':' in p.text]
    return {
        "paragraph_count": len(paragraphs),
        "titles": titles
    }


# 自动化评估
def get_score(generated_doc, reference_doc):
    gen_doc_tokens = word_tokenize(generated_doc)
    ref_doc_tokens = word_tokenize(reference_doc)
    bleu_score = sentence_bleu([ref_doc_tokens], gen_doc_tokens)
    meteor_scores = meteor_score([ref_doc_tokens], gen_doc_tokens)

    return {
        "bleu_score": bleu_score,
        "meteor_score": meteor_scores
    }


def main():
    with open("generation_template.txt", "r", encoding="utf-8") as f:
        generated_doc = f.read()
    with open("reference.txt", "r", encoding="utf-8") as f:
        reference_doc = f.read()

    # 事实性评估
    gen_ents = extract_entities(generated_doc)
    ref_ents = extract_entities(reference_doc)
    fact_check_result = fact_check(gen_ents, ref_ents)

    print("Common entities: ", fact_check_result["common_entities"])
    print("Missing or incorrect entities: ", fact_check_result["missing_or_incorrcet_entities"])
    # 信息性评估
    print("Similarity Score:", get_similarity_score(generated_doc, reference_doc))
    # 篇章结构评估
    generated_analysis = analyze_structure(generated_doc)
    reference_analysis = analyze_structure(reference_doc)

    print("Generated Document Analysis:", generated_analysis)
    print("Reference Document Analysis:", reference_analysis)
    # 自动化评估
    print("BLEU and METEOR Scores:", get_score(generated_doc, reference_doc))



if __name__ == "__main__":
    main()