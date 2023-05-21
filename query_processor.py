import multiprocessing as mp
from pre_processor import pre_process
from data_load import index_decode
from queue import PriorityQueue
from math import log
from time import time

MAX_SIZE = 20


def daat(index, tokens, docids):

    results = PriorityQueue(maxsize=MAX_SIZE)

    for target in docids:

        score = 0

        for term in tokens:
            postings = index[term]

            for (docid, tf) in postings:
                if docid == target:
                    score += tf
                    break

        if results.qsize() == MAX_SIZE:
            removed = results.get()
        results.put((score, target))

    return results.queue


def tf_idf(manager, document, sub_index, query_tokens):

    idfs = []
    for token in query_tokens:
        idf = log(len(manager.doc_info)+1 / float(len(sub_index[token])))
        idfs.append(idf)

    tfs = []
    for token in query_tokens:
        frequency = 0
        for i in sub_index[token]:
            if i[0] == document:
                frequency = i[1]
                break

        doc_len = manager.doc_info.loc[document]['token_count']
        tf = frequency/doc_len

        tfs.append(tf)

    tfidfs = []
    for (idf, tf) in zip(idfs, tfs):
        tfidf = tf*idf
        tfidfs.append(tfidf)

    sum_value = sum(tfidfs)

    return sum_value


def bm_25(manager, document, sub_index, query_tokens):

    k = 1.2
    b = .2
    doc_len = manager.doc_info.loc[document]['token_count']

    idfs = []
    for token in query_tokens:
        idf = log(len(manager.doc_info) + 1 / float(len(sub_index[token])))
        idfs.append(idf)

    tfs = []
    for token in query_tokens:
        frequency = 0
        for i in sub_index[token]:
            if i[0] == document:
                frequency = i[1]
                break

        tf = frequency/doc_len

        tfs.append(tf)

    bm_25_weights = []
    for tf in tfs:
        weight = ((k+1)*tf) / (tf + k*((1-b)+b * doc_len / manager.avg_len))
        bm_25_weights.append(weight)

    scores = []
    for (tf, idf, bm) in zip(tfs, idfs, bm_25_weights):
        score = tf*idf*bm
        scores.append(score)

    final_score = sum(scores)

    return final_score


def ranking(manager, documents, sub_index, rank_method, query_tokens):

    rank = []

    documents = [doc[1] for doc in documents]
    for document in documents:

        if rank_method == 'TFIDF':
            rank_value = tf_idf(manager, document, sub_index, query_tokens)
            rank.append((rank_value, document))
        else:
            rank_value = bm_25(manager, document, sub_index, query_tokens)
            rank.append((rank_value, document))

    rank.sort(reverse=True)

    rank = rank[:10]

    return rank


def query_processing(query, manager, results, rank_method):

    time_start = time()
    query_tokens = pre_process(query)

    sub_index = index_decode(manager.index, manager.lexic, query_tokens)

    match_begin = time()
    common_docs = set()
    for pos, key in enumerate(sub_index.keys()):
        docs = [i[0] for i in sub_index[key]]
        if pos == 0:
            common_docs.update(docs)
        else:
            new_set = set()
            for new_doc in docs:
                if new_doc in common_docs:
                    new_set.add(new_doc)

            common_docs = new_set

    docids = set()
    for key in sub_index.keys():
        docs = [i[0] for i in sub_index[key]]
        docids.update(docs)

    begin_daat = time()
    daat_result = daat(sub_index, query_tokens, common_docs)
    # print(f'{query} daat ran for: {time()-begin_daat}')
    # print(f'{query} matching ran for: {time()-match_begin}')

    begin_rank = time()
    rank = ranking(manager, daat_result, sub_index, rank_method, query_tokens)
    # print(f'{query} rank ran for: {time() - begin_rank}')

    results.append((query, rank))

    print('Thread for ', query, ' ran in: ', time()-time_start, 'seconds')


def multi_processing(index, lexic, doc_info, queries, rank_method):

    manager_object = mp.Manager()
    manager_ = manager_object.Namespace()

    results = manager_object.list()

    manager_.index = index
    manager_.lexic = lexic
    manager_.doc_info = doc_info

    lens = doc_info['token_count'].tolist()

    manager_.avg_len = sum(lens)/len(lens)

    query_list = [(query, manager_, results, rank_method) for query in queries]

    with mp.Pool(4) as pool:
        pool.starmap(query_processing, query_list)

    return results

