# -*- coding: utf-8 -*-

import os
import io
import operator
import pprint

sentence_list = []
with io.open('orchid97.crp.utf', 'r', encoding='utf-8') as dataset:
    with io.open('orchid_sentences.txt', 'w', encoding='utf-8') as output:
        for line in dataset:
            if not (line.startswith('#') or line.startswith('%')) and line.strip().endswith('//') and len(line) > 3:
                output.write(line.strip().strip('//') +'\n')
