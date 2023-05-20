# You can (and must) freely edit this file (add libraries, functions and calls) to implement your query processor
import pandas as pd
from data_load import data_loader, queries_loader
from query_processor import multi_processing
from time import time
import argparse

import warnings
warnings.filterwarnings("ignore")

INDEX_PATH_ = 'C:/Users/Desktop/Desktop/Misc/UFMG/Isolada/23_1/RI/tp2/index/'
QUERY_PATH_ = 'C:/Users/Desktop/Desktop/Misc/UFMG/Isolada/23_1/RI/tp2/queries-sample.txt'


def rank_to_json(rank):

    query = str(rank[0]).replace('\n', '')
    json = '{ "Query": ' + query + ',\n  "Results": ['

    for document in rank[1]:
        json += '\n    { "ID": "' + str(document[1]) + '",\n      "Score": ' + str(document[0]) + ' },'

    json += ' ]}'

    return json


def ranks_to_json(ranks):

    jsons = []
    for rank in ranks:
        jsons.append(rank_to_json(rank))

    return jsons


def main(index_path_, query_path_, ranker_):

    start_time = time()
    index, lexic, doc_info = data_loader(index_path_)
    load_time = time() - start_time

    print('load time: ', load_time)

    queries = queries_loader(query_path_)

    ranks = multi_processing(index, lexic, doc_info, queries, ranker_)
    jsons = ranks_to_json(ranks)

    for json in jsons:
        print(json)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest='index_path', action='store')
    parser.add_argument('-q', type=str, dest='query_path', action='store')
    parser.add_argument('-r', type=str, dest='ranker', action='store')

    args = parser.parse_args()
    index_path = args.index_path
    query_path = args.query_path
    ranker = args.ranker

    main(index_path, query_path, ranker)
