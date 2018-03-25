


import io

sentences_path = 'data/orchid_sentences.txt'

output_path = 'output/orichid_dataset.txt'


with io.open(sentences_path, 'r', encoding='utf-8') as f:
    for line in f:
        if len(line) > 20:
            print(line)
