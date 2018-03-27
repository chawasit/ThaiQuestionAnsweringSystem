# -*- coding: utf-8 -*-

import os
import io
import operator
import pprint
from lexto.LexTo import LexTo
import pathlib
import fire
import glob
import json
import numpy as np
import copy
import functools
import re
import deepcut

lexto = LexTo()

OUTPUT_DIRECTORY = 'output/'

DOCUMENT_DIRECTORY = 'data/Examples'
SOURCE_DIRECTORY = os.path.join(DOCUMENT_DIRECTORY, 'Sources')
SOURCE_LIST_PATH = os.path.join(DOCUMENT_DIRECTORY, 'source_list.txt')
QUESTION_LIST_PATH = os.path.join(DOCUMENT_DIRECTORY, 'question_list.txt')
INTEREST_TOKEN_TYPE = ['NN', 'NR', 'FWN', 'VV']

def write_file(file_path, content):
    file_path = os.path.join(OUTPUT_DIRECTORY, *file_path)
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
    return filter_token_by_type(token_list, INTEREST_TOKEN_TYPE)

def only_word(token):
    return split_token(token)[0]

def json_dump(obj):
    return json.dumps(obj, ensure_ascii=False)

def read_data(path):
    ret = ''
    with io.open(path, 'r', encoding='utf-8-sig') as data:
        ret = data.read()
    return ret

def read_output(path):
    ret = ''
    with io.open(path, 'r', encoding='utf-8') as data:
        ret = data.read()
    return ret

def count_word(text, word):
    return sum(1 for _ in re.finditer(r'%s' % re.escape(word), text))
        
def glob_directory(*directory):
    return sorted(glob.glob(os.path.join(OUTPUT_DIRECTORY, *directory)))

def glob_data(*directory):
    return sorted(glob.glob(os.path.join(DOCUMENT_DIRECTORY, *directory)))

def merge_token(raw_tokens):
    tokens = copy.deepcopy(raw_tokens)
    number_of_token = len(tokens)
    flag = np.zeros(number_of_token)

    def case_nn_nr(index):
        token = tokens[i]
        prev_token = tokens[i-1]
        tag = token_type(token)
        prev_tag = token_type(prev_token)
        true_flag =  flag[i-1] if i-1 >= 0 else False
        return tag in ['NN', 'NR'] \
            and (prev_tag in ['NN', 'NR'] or true_flag)

    def case_p_pu(index):
        return False
        try:
            next_token = tokens[i+1]
            token = tokens[i]
            prev_token = tokens[i-1]
            next_tag = token_type(next_token)
            tag = token_type(token)
            prev_tag = token_type(prev_token)
            return tag in ['P'] \
                and next_tag in ['NN', 'NR'] \
                and prev_tag in ['NN', 'NR']
        except:
            return False

    for i in range(1, number_of_token):
        if case_nn_nr(i) or case_p_pu(i):
            flag[i] = flag[i-1] = 1

    for i in range(number_of_token-2, -1, -1):
        if flag[i] and flag[i+1]:
            base_token = tokens[i]
            join_token = tokens[i+1]
            base_word = only_word(base_token)
            join_word = only_word(join_token)
            merge_word = base_word + join_word
            tokens[i] =  merge_word + "/NE"
            tokens.pop(i+1)

    return tokens

def build_document_dictionary():
    print("Building Dictionary")
    file_list = glob_directory('tokenize', '*.txt')

    corpus_dictionary = {}
    document_dictionary = {}
    for file_path in file_list:
        with io.open(file_path, 'r', encoding='utf-8-sig') as document:
            filename = os.path.basename(file_path)
            dictionary = {}
            token_string = document.read()
            tokens = token_string.split('|')
            tokens = filter_token_by_type(tokens, ['NN', 'NE', 'VV', 'FWN', 'NR', 'OD', 'CL', 'CD'])

            for token in tokens:
                word = only_word(token)
                if word not in corpus_dictionary:
                    corpus_dictionary[word] = 0
                corpus_dictionary[word] += 1

                if word not in dictionary:
                    dictionary[word] = 0
                dictionary[word] += 1

            document_dictionary[filename] = dictionary
    print("Complete")

    return document_dictionary, corpus_dictionary

def build_document_dictionary2():
    print("Building Dictionary")
    
    corpus_dictionary = {}
    document_dictionary = {}
    with io.open(SOURCE_LIST_PATH, 'r') as source_list:
        for file_name in source_list:
            file_name = file_name.strip()
            document_path = os.path.join(SOURCE_DIRECTORY, file_name)
            with io.open(document_path, 'r', encoding='utf-8-sig') as document:
                filename = os.path.basename(document_path)
                dictionary = {}
                text = document.read()
                words = lexto.tokenize(text)[0]
                # words = deepcut.tokenize(text)
                for word in words:
                    if word not in corpus_dictionary:
                        corpus_dictionary[word] = 0
                    corpus_dictionary[word] += 1

                    if word not in dictionary:
                        dictionary[word] = 0
                    dictionary[word] += 1

                document_dictionary[filename] = dictionary
                write_file(['dictionary', filename], json_dump(dictionary))
    print("Complete")

    return document_dictionary, corpus_dictionary

def tokenize_corpus():
    from SynThai import SynThai

    synthai = SynThai('model/SynThai/0063-0.0412.hdf5', 60)

    with io.open(SOURCE_LIST_PATH, 'r') as source_list:
        for file_name in source_list:
            file_name = file_name.strip()
            document_path = os.path.join(SOURCE_DIRECTORY, file_name)
            with io.open(document_path, 'r', encoding='utf-8-sig') as document:
                text = document.read()
                
                tokens_string = synthai.tokenize(text).strip('|')

                write_file(['tokenize', file_name], tokens_string)


def merge_document_tokens():
    print("Merging noun and noun relate")

    file_list = glob_directory('tokenize', "*.txt")
    print("Total files: ", len(file_list))
    for file_path in file_list:
        print(file_path)
        with io.open(file_path, 'r', encoding='utf-8-sig') as document:
            filename = os.path.basename(file_path)
            tokenized_doc = document.read()
            i = 0
            merged_nr = []
            tokens = tokenized_doc.split('|')
            
            merged_tokens = merge_token(tokens)

            merged_string = '|'.join(merged_tokens)

            write_file(['merge_document', filename], merged_string)


def tokenize_question():
    from SynThai import SynThai

    synthai = SynThai('model/SynThai/0063-0.0412.hdf5', 60)

    with io.open(QUESTION_LIST_PATH, 'r', encoding='utf-8-sig') as question_list:
        for line in question_list:
            id, question = line.split('::')

            tokens_string = synthai.tokenize(question).strip('|')
            tokens = tokens_string.split('|')

            write_file(['tokenize_question', id + '.txt'], tokens_string)


def classify_question():
    file_list = glob_directory('tokenize_question', '*.txt')

    for file_path in file_list:
        print(file_path)
        with io.open(file_path, 'r', encoding='utf-8-sig') as tokenize_question:
            qid = os.path.basename(file_path)
            tokens = tokenize_question.read().split('|')


def merge_question_tokens():
    print("Merging noun and noun relate")

    file_list = glob_directory('tokenize_question', "*.txt")
    print("Total files: ", len(file_list))
    for file_path in file_list:
        print(file_path)
        with io.open(file_path, 'r', encoding='utf-8-sig') as document:
            filename = os.path.basename(file_path)
            tokenized_doc = document.read()
            i = 0
            merged_nr = []
            tokens = tokenized_doc.split('|')
            
            merged_tokens = merge_token(tokens)

            merged_string = '|'.join(merged_tokens)

            write_file(['merge_question', filename], merged_string)


def term_frequency(dictionary, word):
    if word not in dictionary:
        return 0.5

    max_frequency = 0
    for word, count in dictionary.items():
        max_frequency = max(max_frequency, count)

    return 0.5 + 0.5 * dictionary[word] / max_frequency


def inverse_document_frequency(corpus_dictionary, word):
    count = 0.1
    for doc, dictionary in corpus_dictionary.items():
        if word in dictionary:
            count += 1.0
    
    return np.log(len(corpus_dictionary) / count)

def calculate_tf_idf(corpus_dictionary, document_dictionary, word):
    return term_frequency(document_dictionary, word) * inverse_document_frequency(corpus_dictionary, word) 

def term_frequency2(dictionary, word):
    if word not in dictionary:
        return 0.5

    max_frequency = 0
    for word, count in dictionary.items():
        max_frequency = max(max_frequency, count)

    return 0.5 + 0.5 * dictionary[word] / max_frequency

def inverse_document_frequency2(corpus_dictionary, word):
    count = 0.1
    
    for doc, dictionary in corpus_dictionary.items():
        if word in dictionary:
            count += 1.0
    
    return np.log(len(corpus_dictionary) / count)

def calculate_tf_idf2(corpus_dictionary, document_dictionary, word):
    return term_frequency2(document_dictionary, word) * inverse_document_frequency2(corpus_dictionary, word) 


def rank_document():
    document_dictionary, corpus_dictionary = build_document_dictionary()

    file_list = glob_directory('tokenize_question', '*.txt')

    
    for file_path in file_list:
        filename = os.path.basename(file_path)

        with io.open(file_path, 'r', encoding='utf-8-sig') as tokenize_question:
            qid = os.path.basename(file_path)
            tokens = tokenize_question.read().split('|')
            tokens = filter_token_by_type(tokens, ['NN', 'NE', 'VV', 'FWN', 'NR', 'OD', 'CL', 'CD'])
            words = [only_word(token) for token in tokens]
            document_relevent_score = []
            for document_name, dictionary in document_dictionary.items():
                tf_idfs = [calculate_tf_idf(document_dictionary, dictionary, word) for word in words]
                score = sum(tf_idfs)
                document_relevent_score.append( (document_name, score) )

            print(filename, sorted(document_relevent_score, key=operator.itemgetter(1), reverse=True)[:5])


def rank_document2():
    document_dictionary, corpus_dictionary = build_document_dictionary2()

    with io.open(QUESTION_LIST_PATH, 'r', encoding='utf-8-sig') as question_list:
        for line in question_list:
            id, question = line.split('::')

            words = lexto.tokenize(question)[0]
            # words = deepcut.tokenize(question)
            document_relevent_score = []
            for document_name, dictionary in document_dictionary.items():
                tf_idfs = [calculate_tf_idf(document_dictionary, dictionary, word) for word in words]
                score = sum(tf_idfs)
                document_relevent_score.append( (document_name, score) )

            print(id, sorted(document_relevent_score, key=operator.itemgetter(1), reverse=True)[:5])
            

if __name__ == "__main__":
    fire.Fire({
        "tokenize": tokenize_corpus,
        "merge_document": merge_document_tokens,
        "tokenize_question": tokenize_question,
        "merge_question": merge_question_tokens,
        "rank_document": rank_document,
        "rank_document2": rank_document2

    })
