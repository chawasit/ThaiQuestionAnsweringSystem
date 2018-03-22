# -*- coding: utf-8 -*-

from SynThai import SynThai
import os
import io
import operator
import pprint
from lexto.LexTo import LexTo

lexto = LexTo()

# output_directory = 'output/lexto'
output_directory = 'output/synthai'

document_directory = 'data/Examples'
source_directory = os.path.join(document_directory, 'Sources')
source_list_path = os.path.join(document_directory, 'source_list.txt')
interesting_token_type = ['NN', 'NR', 'FWN']

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


synthai = SynThai('model/SynThai/0063-0.0412.hdf5', 60)

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
            # tokens = lexto.tokenize(text)[0]
            for token in tokens:
                if token not in corpus_dictionary:
                    corpus_dictionary[token] = 0
                corpus_dictionary[token] += 1

                if token not in document_token_list:
                    document_token_list.append(token)
            
            write_file('document_' + file_name.strip(), pprint.pformat(document_token_list)) 
            

corpus_token_list = sorted(corpus_dictionary.items(), key=operator.itemgetter(1), reverse=True)
write_file('corpus.txt', pprint.pformat(corpus_token_list))

print(len(corpus_token_list))

question_path = os.path.join(document_directory, 'question_list.txt')
with io.open(question_path, 'r', encoding='utf-8-sig') as question_file:
    for question_line in question_file:
        id, question = question_line.strip().split('::')
        question_token_list = []
        tokens_string = synthai.tokenize(question)
        tokens = tokens_string.split('|')
        tokens = filter_interest_token(tokens)
        # tokens = lexto.tokenize(question)[0]
        for token in tokens:
            if token not in question_token_list:
                question_token_list.append(token)
        
        write_file('question_' + id + '.txt', pprint.pformat(question_token_list)) 

