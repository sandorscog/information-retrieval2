import pandas as pd


def index_decode(byte_string, lexic, tokens):

    index = {}

    # iterates over the words
    for token, row in lexic.loc[tokens].iterrows():
        # reads each of the occurrences of the word
        entry_point = row['entry_point']
        occurrence_list = []
        for position in range(row['size']):
            init = (position * 8) + entry_point
            docid = int.from_bytes(byte_string[init: init+4], 'big')
            tf = int.from_bytes(byte_string[init+4: init+8], 'big')

            occurrence_list.append((docid, tf))

        occurrence_list.sort()
        index[token] = occurrence_list

    return index


def data_loader(path):
    # Loading all index structures
    with open(path + 'lexic.txt', 'r', encoding='utf8') as f:
        lexic = f.readlines()
        lexic = [i.replace('\n', '') for i in lexic]
        lexic = [eval(i) for i in lexic]
        lexic = pd.DataFrame(lexic, columns=['token', 'entry_point', 'size'])
        lexic.set_index('token', inplace=True)

    with open(path + 'index', 'rb') as f:
        index = f.read()
        # index = index_decode(index, lexic)

    doc_info = pd.read_csv(path + 'doc_index.csv')

    return index, lexic, doc_info


def queries_loader(path):
    with open(path, 'r') as f:
        queries = f.readlines()
    return queries


