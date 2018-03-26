# -*- coding: utf-8 -*-

from SynThai import SynThai
import os
import io
import operator
import pprint
from lexto.LexTo import LexTo
import pathlib

lexto = LexTo()

output_directory = 'output/synthai'

document_directory = 'data/Examples'
source_directory = os.path.join(document_directory, 'Sources')
source_list_path = os.path.join(document_directory, 'source_list.txt')
interesting_token_type = ['NN', 'NR', 'FWN', 'VV`']

def write_file(file_path, content):
    file_path = os.path.join(output_directory, file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with io.open(file_path, 'w', encoding='utf8') as file:
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

synthai = SynThai('model/SynThai/0063-0.0412.hdf5', 60)

corpus_raw_documents = {}
corpus_dictionary = {}
with io.open(source_list_path, 'r') as source_list:
    for file_name in source_list:
        document_path = os.path.join(source_directory, file_name.strip())
        with io.open(document_path, 'r', encoding='utf-8-sig') as document:
            document_token_list = []
            text = document.read()
            
            tokens_string = synthai.tokenize(text)
            corpus_raw_documents[file_name.strip()] = tokens_string
            tokens = tokens_string.split('|')
            # tokens = filter_interest_token(tokens)
            # tokens = lexto.tokenize(text)[0]
            for token in tokens:
                if token not in corpus_dictionary:
                    corpus_dictionary[token] = 0
                corpus_dictionary[token] += 1

                if token not in document_token_list:
                    document_token_list.append(token)
            write_file('docs/'+file_name.strip(), tokens_string)

merged_noun_corpus = {}
for key, raw in corpus_raw_documents.items():
    print(raw)
    i = 0
    merged_nr = []
    tokens = raw.split('|')
    while i < len(tokens):
        token = tokens[i]
        word = only_word(token)
        tag = token_type(token)
        if tag == "NN" or tag == "NR":
            while i < len(tokens):
                token = tokens[i]
                word = only_word(token)
                tag = token_type(token)
                if tag != "NR":
                    merged_nr.append('/NN')
                    merged_nr.append(token)
                    break
                merged_nr.append(word)
                i += 1
        else:
            merged_nr.append(token)
        merged_nr.append('|')
        i += 1
    merged_noun_corpus[key] = "".join(merged_nr)

    write_file('merge/'+key, merged_noun_corpus[key])
