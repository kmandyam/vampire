import argparse
import json
import os
from typing import List, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from vampire.common.util import read_text, save_sparse, write_to_json

METADATA = ['in_reply_to_status_id_str', 'in_reply_to_status_id', 'lang', 'source', 'in_reply_to_user_id_str',
            'in_reply_to_screen_name', 'favorited', 'created_at', 'is_quote_status', 'in_reply_to_user_id',
            'contributors', 'retweeted', 'place', 'favorite_count', 'retweet_count', 'truncated']

def load_data(data_path: str, load_covars=False) -> Tuple[List[str], List[str]]:
    tokenized_examples = []
    tokenized_covars = []
    with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
        for line in f:
            if data_path.endswith(".jsonl") or data_path.endswith(".json"):
                example = json.loads(line)
            else:
                example = {"text": line}
            text = example['text']
            tokenized_examples.append(text)
            if load_covars:
                covariates = []
                for c in METADATA:
                    covariates.append(example[c])
                tokenized_covars.append(' '.join(covariates))
    return tokenized_examples, tokenized_covars

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to the train jsonl file.")
    parser.add_argument("--dev-path", type=str, required=True,
                        help="Path to the dev jsonl file.")
    parser.add_argument("--serialization-dir", "-s", type=str, required=True,
                        help="Path to store the preprocessed output.")
    parser.add_argument("--tfidf", action='store_true',
                        help="use TFIDF as input")
    parser.add_argument("--vocab-size", type=int, required=False, default=10000,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenizer-type", type=str, default="just_spaces",
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--reference-corpus-path", type=str, required=False,
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--tokenize-reference", action='store_true',
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--reference-tokenizer-type", type=str, default="just_spaces",
                        help="Path to store the preprocessed corpus vocabulary (output file name).")
    parser.add_argument("--preprocess-covariates", type=bool, default=True,
                        help="Whether we want to preprocess the metadata associated with each example")
    args = parser.parse_args()

    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)

    vocabulary_dir = os.path.join(args.serialization_dir, "vocabulary")

    if not os.path.isdir(vocabulary_dir):
        os.mkdir(vocabulary_dir)

    tokenized_train_examples, _ = load_data(args.train_path, load_covars=False)
    tokenized_dev_examples, _ = load_data(args.dev_path, load_covars=False)

    if args.preprocess_covariates:
        tokenized_train_examples, tokenized_train_covariates = load_data(args.train_path, load_covars=True)
        tokenized_dev_examples, tokenized_dev_covariates = load_data(args.dev_path, load_covars=True)

    print("fitting count vectorizer...")
    if args.tfidf:
        count_vectorizer = TfidfVectorizer(stop_words='english', max_features=args.vocab_size,
                                           token_pattern=r'\b[^\d\W]{3,30}\b')
        covar_vectorizer = TfidfVectorizer(stop_words='english', max_features=args.vocab_size,
                                           token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        count_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size,
                                           token_pattern=r'\b[^\d\W]{3,30}\b')
        covar_vectorizer = CountVectorizer(stop_words='english', max_features=args.vocab_size,
                                           token_pattern=r'\b[^\d\W]{3,30}\b')

    text = tokenized_train_examples + tokenized_dev_examples

    count_vectorizer.fit(tqdm(text))

    vectorized_train_examples = count_vectorizer.transform(tqdm(tokenized_train_examples))
    vectorized_dev_examples = count_vectorizer.transform(tqdm(tokenized_dev_examples))

    if args.preprocess_covariates:
        covar_text = tokenized_train_covariates + tokenized_dev_covariates

        covar_vectorizer.fit(tqdm(covar_text))

        vectorized_train_covariates = covar_vectorizer.transform(tqdm(tokenized_train_covariates))
        vectorized_dev_covariates = covar_vectorizer.transform(tqdm(tokenized_dev_covariates))

    if args.tfidf:
        reference_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
        reference_covariate_vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    else:
        reference_vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
        reference_covariate_vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[^\d\W]{3,30}\b')
    if not args.reference_corpus_path:
        print("fitting reference corpus using development data...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(tokenized_dev_examples))
        print("fitting reference corpus covariates using development data")
        if args.preprocess_covariates:
            reference_covariate_matrix = reference_covariate_vectorizer.fit_transform(tqdm(tokenized_dev_covariates))
    else:
        print(f"loading reference corpus at {args.reference_corpus_path}...")
        reference_examples = load_data(args.reference_corpus_path, args.tokenize_reference,
                                       args.reference_tokenizer_type)
        print("fitting reference corpus...")
        reference_matrix = reference_vectorizer.fit_transform(tqdm(reference_examples))
        if args.preprocess_covariates:
            reference_covariate_matrix = reference_covariate_vectorizer.fit_transform(tqdm(tokenized_dev_covariates))

    reference_vocabulary = reference_vectorizer.get_feature_names()
    if args.preprocess_covariates:
        reference_covariate_vocabulary = reference_covariate_vectorizer.get_feature_names()

    # add @@unknown@@ token vector
    vectorized_train_examples = sparse.hstack(
        (np.array([0] * len(tokenized_train_examples))[:, None], vectorized_train_examples))
    vectorized_dev_examples = sparse.hstack(
        (np.array([0] * len(tokenized_dev_examples))[:, None], vectorized_dev_examples))
    master = sparse.vstack([vectorized_train_examples, vectorized_dev_examples])

    if args.preprocess_covariates:
        vectorized_train_covariates = sparse.hstack(
            (np.array([0] * len(tokenized_train_covariates))[:, None], vectorized_train_covariates))
        vectorized_dev_covariates = sparse.hstack(
            (np.array([0] * len(tokenized_dev_covariates))[:, None], vectorized_dev_covariates))
        covar_master = sparse.vstack([vectorized_train_covariates, vectorized_dev_covariates])

    # generate background frequency
    print("generating background frequency...")
    bgfreq = dict(zip(count_vectorizer.get_feature_names(), (np.array(master.sum(0)) / args.vocab_size).squeeze()))

    if args.preprocess_covariates:
        covar_bgfreq = dict(
            zip(covar_vectorizer.get_feature_names(), (np.array(covar_master.sum(0)) / args.vocab_size).squeeze()))

    print("saving data...")
    save_sparse(vectorized_train_examples, os.path.join(args.serialization_dir, "train.npz"))
    save_sparse(vectorized_dev_examples, os.path.join(args.serialization_dir, "dev.npz"))
    if not os.path.isdir(os.path.join(args.serialization_dir, "reference")):
        os.mkdir(os.path.join(args.serialization_dir, "reference"))
    save_sparse(reference_matrix, os.path.join(args.serialization_dir, "reference", "ref.npz"))
    write_to_json(reference_vocabulary, os.path.join(args.serialization_dir, "reference", "ref.vocab.json"))
    write_to_json(bgfreq, os.path.join(args.serialization_dir, "vampire.bgfreq"))

    write_list_to_file(['@@UNKNOWN@@'] + count_vectorizer.get_feature_names(),
                       os.path.join(vocabulary_dir, "vampire.txt"))
    write_list_to_file(['*tags', '*labels', 'vampire'], os.path.join(vocabulary_dir, "non_padded_namespaces.txt"))

def write_list_to_file(ls, save_path):
    """
    Write each json object in 'jsons' as its own line in the file designated by 'save_path'.
    """
    # Open in appendation mode given that this function may be called multiple
    # times on the same file (positive and negative sentiment are in separate
    # directories).
    out_file = open(save_path, "w+")
    for example in ls:
        out_file.write(example)
        out_file.write('\n')

if __name__ == '__main__':
    main()
