

import io, os
from SynThai import SynThai
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from lexto.LexTo import LexTo
import deepcut
from gensim.models import KeyedVectors
import numpy as np
from scipy import spatial
import operator


word2vec = KeyedVectors.load_word2vec_format('model/thai2vec.vec', binary=False)

document_directory = 'data/Examples'

output_directory = 'output/orchid_sentence_segmentation'
source_directory = os.path.join(document_directory, 'Sources')
source_list_path = os.path.join(document_directory, 'source_list.txt')
interesting_token_type = ['NN', 'NR', 'FWN', 'VV']

def write_file(file_path, content):
    with io.open(os.path.join(output_directory, file_path), 'w', encoding='utf8') as file:
        file.write(content)

def split_token(token):
    return token.split('/')

def token_type(token):
    if len(split_token(token)) < 2:
        return 'NN'
    return split_token(token)[1]

def filter_token_by_type(token_list, type_list):
    return [token for token in token_list if token_type(token) in type_list]

def filter_interest_token(token_list):
    return filter_token_by_type(token_list, interesting_token_type)

def only_word(token):
    return split_token(token)[0]

def create_sentence_vector(words):
    vector = np.zeros((1, 300))
    for word in words:
        if word in word2vec.wv.index2word:
            vector += word2vec.wv.word_vec(word)
    vector = vector / len(words)
    return vector

synthai = SynThai('model/SynThai/0063-0.0412.hdf5', 60)

# lexto = LexTo()

document_vectors = []
documents = []
corpus_dictionary = {}
with io.open(source_list_path, 'r') as source_list:
    for file_name in source_list:
        document_path = os.path.join(source_directory, file_name.strip())
        with io.open(document_path, 'r', encoding='utf-8-sig') as document:
            document_token_list = []
            text = document.read()
            tokens_string = synthai.tokenize(text)
            tokens = tokens_string.split('|')
            tokens = filter_interest_token(tokens)
            words = [only_word(token) for token in tokens]
            # words = lexto.tokenize(text)[0]
            # words = deepcut.tokenize(text)
            # print(file_name, doc_vector)
            documents.append(
                LabeledSentence(words=words, tags=['doc_' + file_name.split('.')[0]])
            )

# model = Doc2Vec(documents, vector_size=4000, window=4, min_count=1, workers=8)
# model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
# model.save('output/pedia.dec2vec')


with io.open(source_list_path, 'r') as source_list:
    for file_name in source_list:
        document_path = os.path.join(source_directory, file_name.strip())
        with io.open(document_path, 'r', encoding='utf-8-sig') as document:
            document_token_list = []
            text = document.read()
            # tokens_string = synthai.tokenize(text)
            # tokens = tokens_string.split('|')
            # tokens = filter_interest_token(tokens)
            # words = [only_word(token) for token in tokens]
            # words = lexto.tokenize(text)[0]
            words = deepcut.tokenize(text)
            # question_vector = model.infer_vector(words)
            # sims = model.docvecs.most_similar([question_vector])
            # print(file_name, sims)

            distances.sort(key=operator.itemgetter(1), reverse=False)
            print(id, distances[:5])

question_path = os.path.join(document_directory, 'question_list.txt')
with io.open(question_path, 'r', encoding='utf-8-sig') as question_file:
    for question_line in question_file:
        id, question = question_line.strip().split('::')
        question_token_list = []
        tokens_string = synthai.tokenize(question)
        tokens = tokens_string.split('|')
        tokens = filter_interest_token(tokens)
        words = [only_word(token) for token in tokens]
        
        # words = lexto.tokenize(question)[0]
        # words = deepcut.tokenize(question)
        # question_vector = model.infer_vector(words)
        # sims = model.docvecs.most_similar([question_vector])
        # print(file_name, sims)
