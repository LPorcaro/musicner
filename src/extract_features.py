#!/usr/bin/env python
# encoding: utf-8

import io
import argparse
import logging
from backports import csv

from pos_chunk_twitter_nlp import PosChunkTagger
from utils import import_config, set_log_config


class ExtractFeatures(object):
    """
    It extracts several features from the input tweets for performing
    experiments with Neural Networks and Machine Learning models. It creates
    two output files, one which can be used as input in WEKA, and one which can
    be used as input in BiLSTM-CNN-CRF architecture for sequence tagging
    implementation https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf

    Usage:
    python extract_features.py
    -i ../path/to/INPUTFILE_summary.csv
    -e ../path/to/INPUTFILE_entities.csv
    -o ../path/to/OUTPUTFILE_WEKA.csv
    -n ../path/to/OUTPUTFILE_NeuralNetworks.csv
    """

    def __init__(self, input_file, input_ent, output_weka, output_nn, limit):
        """
        """
        self.input_file = input_file
        self.input_ent = input_ent
        self.out_weka = output_weka
        self.out_nn = output_nn
        self.limit = limit

        self.cfg_feat = import_config('features')
        self.gazzetters = self.import_gazetters()
        self.DictEntities = self.import_entities_annotated()
        self.tagger = PosChunkTagger()

        self.POSTags = set()
        self.ChunkTags = set()

        self.out_tokens = []

    def write_weka(self):
        """
        Write the file for performing experiments with WEKA
        """
        with io.open(
                self.out_weka, 'w+', newline='', encoding='utf-8') as outf:
            self.write_weka_header(outf)
            self.write_weka_data(outf)

    def write_weka_header(self, outf):
        """
        Write header of WEKA file
        """
        NULL_EL = ['NULL']
        POSTags = ",".join(
            ["'%s'" % x for x in self.POSTags
                if not x.startswith("'")] + NULL_EL)
        ChunkTags = ",".join(
            ["'%s'" % x for x in self.ChunkTags
                if not x.startswith("'")] + NULL_EL)

        numeric = 'NUMERIC'
        string_attr = 'string'
        boolean = '{f,t}'
        eclasses = '{O, B-Contributor, I-Contributor, B-Work, I-Work}'

        header = [
            '@relation ner',
            '\n@attribute token %s' % string_attr,
            '\n@attribute eclass %s' % eclasses,
            '\n@attribute POStag {%s}' % POSTags,
            '\n@attribute chunk {%s}' % ChunkTags,
            '\n@attribute position %s' % numeric,
            '\n@attribute iscapital %s ' % boolean,
            '\n@attribute isidigit %s ' % boolean,
            '\n@attribute isfirstname %s ' % boolean,
            '\n@attribute islastname %s ' % boolean,
            '\n@attribute iscontrtype %s ' % boolean,
            '\n@attribute isinstrument %s ' % boolean,
            '\n@attribute isworktype %s ' % boolean,
            '\n@attribute isnoteswork %s ' % boolean,
            '\n@attribute ismodework %s ' % boolean,
            '\n@attribute isopuswork %s ' % boolean,
            '\n@attribute isnumberwork %s ' % boolean,
            '\n@attribute token-2 %s ' % string_attr,
            '\n@attribute token-2-POStag {%s} ' % POSTags,
            '\n@attribute token-2-chunk {%s} ' % ChunkTags,
            '\n@attribute token-1 %s ' % string_attr,
            '\n@attribute token-1-POStag {%s} ' % POSTags,
            '\n@attribute token-1-chunk {%s}' % ChunkTags,
            '\n@attribute token+1 %s ' % string_attr,
            '\n@attribute token+1-POStag {%s} ' % POSTags,
            '\n@attribute token+1-chunk {%s}' % ChunkTags,
            '\n@attribute token+2 %s ' % string_attr,
            '\n@attribute token+2-POStag {%s} ' % POSTags,
            '\n@attribute token+2-chunk {%s} ' % ChunkTags,
            '\n\n@data\n'
            ]

        for el in header:
            outf.write(unicode(el))

    def write_weka_data(self, outf):
        """
        Write data part of WEKA file
        """
        for token in self.out_tokens:
            token = list(token)
            # Remove tweet_id from features
            del token[16]
            # Sanity Check over token feature vectorlength
            if len(token) != 28:
                continue
            # Hacks for assuring WEKA file integrity
            if token[0] in [',', '.'] or not token[0]:
                continue
            for idx in [0, 16, 19, 22, 25]:
                if "'" in token[idx]:
                    token[idx] = token[idx].replace("'", '"')
                if token[idx] != 'NULL':
                    token[idx] = "'%s'" % token[idx]
            for idx in [2, 17, 20, 23, 26]:
                if token[idx] == "''" or token[idx] == ",":
                    token[idx] = ":"

            outf.write("%s\n" % ','.join([unicode(x) for x in token]))

    def write_NN(self):
        """
        Write the file for performing experiments with Neural Networks
        """
        tweet_id_tmp = None
        with io.open(self.out_nn, 'w+', newline='', encoding='utf-8') as outf:
            for token in self.out_tokens:
                token = list(token)
                tweet_id = token[16]
                # Remove tweet_id from features
                del token[16]
                # Sanity Check over token feature vector length
                if len(token) != 28:
                    continue
                # Print blank line after tweet ends
                if tweet_id != tweet_id_tmp and not tweet_id_tmp:
                    tweet_id_tmp = tweet_id
                elif tweet_id != tweet_id_tmp:
                    tweet_id_tmp = tweet_id
                    outf.write(unicode('\n'))

                outf.write("%s\n" % '\t'.join([unicode(x) for x in token]))

    def normalize_position(self, max_len):
        """
        Normalize token position using the max lenght of a tweet
        """
        for i, token in enumerate(self.out_tokens):
            token = list(token)
            token[4] = token[4]/float(max_len)
            token = tuple(token)
            self.out_tokens[i] = token

    def get_contextual_features(self):
        """
        For each token in the input, it extracts several contextual features,
        using the information from the previous and following two tokens.
        """
        null_tuple = ('NULL', 'NULL', 'NULL', )
        for i, token in enumerate(self.out_tokens):

            # Token -2
            if (self.out_tokens[i][4] - 2 == (self.out_tokens[i-2][4]) and
                    self.out_tokens[i-2]):
                self.out_tokens[i] += (self.out_tokens[i-2][0], ) +\
                                       self.out_tokens[i-2][2:4]
            else:
                self.out_tokens[i] += null_tuple

            # Token -1
            if (self.out_tokens[i][4] - 1 == (self.out_tokens[i-1][4]) and
                    self.out_tokens[i-1]):
                self.out_tokens[i] += (self.out_tokens[i-1][0], ) +\
                                       self.out_tokens[i-1][2:4]
            else:
                self.out_tokens[i] += null_tuple

            # Token +1
            if i+1 < len(self.out_tokens):
                if (self.out_tokens[i][4] + 1 == (self.out_tokens[i+1][4]) and
                        self.out_tokens[i+1]):
                    self.out_tokens[i] += (self.out_tokens[i+1][0], ) +\
                                           self.out_tokens[i+1][2:4]
                else:
                    self.out_tokens[i] += null_tuple
            else:
                self.out_tokens[i] += null_tuple

            # Token +2
            if i+2 < len(self.out_tokens):
                if (self.out_tokens[i][4] + 2 == (self.out_tokens[i+2][4]) and
                        self.out_tokens[i+2]):
                    self.out_tokens[i] += (self.out_tokens[i+2][0], ) +\
                                           self.out_tokens[i+2][2:4]
                else:
                    self.out_tokens[i] += null_tuple
            else:
                self.out_tokens[i] += null_tuple

    def get_boolean_features(self):
        """
        For each token in the input, it extracts several boolean features
        """
        true_token = ('t',)
        false_token = ('f',)

        for i, token in enumerate(self.tokens_tagged):
            # Add token position
            token = token + (i+1, )
            # Add POS and Chunk tag for WEKA header
            self.POSTags.add(token[2])
            self.ChunkTags.add(token[3])

            # Check if token start with a capital letter
            if token[0][0].isupper():
                token += true_token
            else:
                token += false_token

            # Check if token is a digit
            if token[0].isdigit():
                token += true_token
            else:
                token += false_token

            # Check if token is part of a gazetters
            for gazzetter in self.gazzetters:
                if (token[0].lower() in gazzetter[1] or
                        token[0] in gazzetter[1]):
                    token += true_token
                else:
                    token += false_token

            self.tokens_tagged[i] = token

    def import_gazetters(self):
        """
        Create a set for each gazeetter defined in the config file
        """
        logging.info('Importing Gazeetters...')
        firstnames = set()
        lastnames = set()
        instrtypes = set()
        voicetypes = set()
        worktypes = set()
        notes = set()
        modes = set()
        opus = set()
        numbers = set()

        gazzetters = [(self.cfg_feat['FIRST_NAMES_GAZ'], firstnames),
                      (self.cfg_feat['LAST_NAMES_GAZ'], lastnames),
                      (self.cfg_feat['CONTR_TYPES_GAZ'], voicetypes),
                      (self.cfg_feat['INTRUMENT_TYPES_GAZ'], instrtypes),
                      (self.cfg_feat['WORK_TYPES_GAZ'], worktypes),
                      (self.cfg_feat['NOTES_GAZ'], notes),
                      (self.cfg_feat['MODES_GAZ'], modes),
                      (self.cfg_feat['OPUS_GAZ'], opus),
                      (self.cfg_feat['NUMBER_GAZ'], numbers)]

        for g_file, g_set in gazzetters:
            with io.open(
                    g_file, newline='', encoding='utf-8') as csvfile:
                _reader = csv.reader(csvfile)
                for row in _reader:
                    g_set.add(row[0].lower())

        logging.info('Done!')
        return gazzetters

    def import_entities_annotated(self):
        """
        Import in a Dict the entity annotated, using as key the tweet ID
        """
        logging.info('Importing Annotated Entities...')
        DictEntities = {}

        with io.open(self.input_ent) as inf:
            _reader = csv.reader(inf)
            next(_reader)
            for line in _reader:
                tweet_id, ent, i, e, iob, etype = line
                if tweet_id not in DictEntities:
                    DictEntities[tweet_id] = []
                DictEntities[tweet_id].append((ent, i, e, iob, etype))

        logging.info('Done!')
        return DictEntities

    def get_entities_annotated(self, tweet_id, text):
        """
        Given the id and the text of a tweet, add to each
        token the entities annotated. In case no entity
        is annotated it add the outside tag to the token.
        """
        if tweet_id in self.DictEntities:
            # Create list of entitites tuples (token, entity type)
            ent_tuple_list = self.DictEntities[tweet_id]
            # Assign Entity Type
            ent_count = 0
            for i, token in enumerate(self.tokens_tagged):
                ent_found = False
                for ent in ent_tuple_list[ent_count:]:
                    # Update Entity found
                    if token[0] == ent[0]:
                        self.update_ent(token, ent, i)
                        ent_count += 1
                        ent_found = True
                        break

                if not ent_found:
                    self.update_other_ent(token, i)

        else:
            for i, token in enumerate(self.tokens_tagged):
                self.update_other_ent(token, i)

    def update_ent(self, token, ent, i):
        """
        Add entity tag to token
        """
        token = (token[0],) + ('-'.join(ent[3:]),) + token[1:]
        self.tokens_tagged[i] = token

    def update_other_ent(self, token, i):
        """
        Add outside (O) tags to the token
        """
        token = (token[0],) + ('O',) + token[1:]
        self.tokens_tagged[i] = token

    def run(self):
        """
        Iterate over the input tweets and for each one extract several
        features. It writes the two files needed for the NER experiments
        """
        count = 0
        max_len = 0

        with io.open(self.input_file, newline='', encoding='utf-8') as inf:
            _reader = csv.reader(inf)
            next(_reader)

            # Iterate over User Generate Tweets
            for row in _reader:
                tweet_id, creation_date, text = row[0:3]

                # Break if limit is reached
                if self.limit and count > self.limit-1:
                    break
                if count == 0:
                    logging.info("Processing tweets...")
                elif count % 250 == 0:
                    logging.info("Processed %d tweets", count)
                count += 1

                # Extract POS and Chunk TAG
                self.tokens_tagged = self.tagger.tag_sentence(text)

                # Get max tweet lenght for normalization
                if len(self.tokens_tagged) > max_len:
                    max_len = len(self.tokens_tagged)

                # Add Entities annotations to tokens
                self.get_entities_annotated(tweet_id, text)

                # Add Boolean features to tokens
                self.get_boolean_features()

                # Add tweet_id
                for i, token_tagged in enumerate(self.tokens_tagged):
                    self.tokens_tagged[i] = token_tagged + (tweet_id,)

                self.out_tokens += self.tokens_tagged

            logging.info("Processed %d tweets", count)
            logging.info("Done!")

            # Add contextual features to tokens
            self.get_contextual_features()
            self.normalize_position(max_len)

            # Write output files
            self.write_weka()
            self.write_NN()


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with tweets summary information")
    parser.add_argument("-e", "--input-entities", type=str, dest='input_ent',
                        help="Input file with entitites annotated")
    parser.add_argument("-o", "--output-weka", type=str, dest='output_weka',
                        help="Output Weka file path")
    parser.add_argument("-n", "--output-nn", type=str, dest='output_nn',
                        help="Output Neural Network file path")
    parser.add_argument("-l", "--logfile", type=str,
                        help="Log file path")
    parser.add_argument("-L", "--limit", type=int, dest='limit',
                        help="Limit number of tweet to process")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg_parser()
    set_log_config(args.logfile, logging.INFO)

    ef = ExtractFeatures(args.input_file,
                         args.input_ent,
                         args.output_weka,
                         args.output_nn,
                         args.limit)

    ef.run()
