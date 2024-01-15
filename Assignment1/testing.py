# import nltk
# nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize
import numpy as np
from glove import Glove, Corpus
from scipy.sparse import coo_matrix
# Build the vocabulary. The choices of tokenizer, vocabulary size, context window size are up to you.
def build_corpus(): 
    num = 0
    temp_corpus = []
    with open("output.txt", 'r', encoding='utf-8') as f:
        for line in f:
            #if num > vocubulary_size:
            #   break
            sentence = sent_tokenize(line)
            temp_corpus.extend(sentence)
            num += 1

    corpus = []
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    for para in temp_corpus:
        cutwords1 = word_tokenize(para)
        cutwords2 = [word for word in cutwords1 if word not in interpunctuations]
        corpus.append(cutwords2)

    return corpus

# Follow the experimental setting of Table 1
def compute_probability(corpus, window_size):
    corpus_model = Corpus()
    corpus_model.fit(corpus, window=window_size)
    co_occurance_matrix = corpus_model.matrix.tocsr()
    word2ind = corpus_model.dictionary

    context_words = ["solid", "gas", "water", "fashion"]
    index_ice = word2ind["ice"]
    X_ice = np.sum(co_occurance_matrix[index_ice])
    index_steam = word2ind["steam"]
    X_steam = np.sum(co_occurance_matrix[index_steam])

    for word in context_words:
        index_k = word2ind[word]
        X_ice_k = co_occurance_matrix[index_ice, index_k]
        X_steam_k = co_occurance_matrix[index_steam, index_k]
        P_k_ice = X_ice_k / X_ice
        P_k_steam = X_steam_k / X_steam
        print("P({}|ice) = {}".format(word, P_k_ice))
        print("P({}|steam) = {}".format(word, P_k_steam))
        print("P({}|ice)/P({}|steam) = {}".format(word, word, P_k_ice / P_k_steam))

    return corpus_model

#Train word vectors using Glove. 
def train_glove(corpus_model):
    glove_model = Glove(no_components=100, learning_rate=0.05)
    glove_model.fit(corpus_model.matrix, epochs=30, no_threads=1, verbose=True)
    glove_model.add_dictionary(corpus_model.dictionary)

    return glove_model

#compute cosine similarity
def cos_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / np.sqrt(np.sum(vec1**2) * np.sum(vec2**2))


#main
if __name__ == "__main__":
    # voc_size = 100000
    window_size = 10
    corpus = build_corpus()
    corpus_model = compute_probability(corpus, window_size)
    glove_model = train_glove(corpus_model)
    print('-'*100)
    #results1
    print(glove_model.most_similar("physics", number=2))
    print(glove_model.most_similar("north", number=2))
    print(glove_model.most_similar("queen", number=2))
    print(glove_model.most_similar("car", number=2))
    print('-'*100)
    #result2
    France_vec = glove_model.word_vectors[glove_model.dictionary['France']]
    Spain_vec = glove_model.word_vectors[glove_model.dictionary['Spain']]
    tree_vec = glove_model.word_vectors[glove_model.dictionary['tree']]
    water_vec = glove_model.word_vectors[glove_model.dictionary['water']]
    sky_vec = glove_model.word_vectors[glove_model.dictionary['sky']]
    bird_vec = glove_model.word_vectors[glove_model.dictionary['bird']]
    print("Cosine similarity(France vs Spain):", cos_similarity(France_vec, Spain_vec))
    print("Cosine similarity(tree vs water):", cos_similarity(tree_vec, water_vec))
    print("Cosine similarity(water vs sky):", cos_similarity(water_vec, sky_vec))
    print("Cosine similarity(sky vs bird):", cos_similarity(sky_vec, bird_vec))
    print('-'*100)
    #result3
    print(glove_model.most_similar("text", number=6))
    
