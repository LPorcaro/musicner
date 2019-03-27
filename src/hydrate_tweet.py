#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import json
import io
import twitter
import argparse
import logging
import time
import twitter_nlp.python.twokenize as twk

from twitter.error import TwitterError
from tqdm import tqdm
from backports import csv

from utils import import_config, set_log_config


class HydrateAnnotated(object):
    """
    It retrieves informations from the annotated tweets corpora and create
    three files:
    1) INPUTFILE_entities.csv: file with information about entities annotated
    2) INPUTFILE_summary.csv: file with information about tweets
    3) INPUTFILE__text_tkn.txt: file with tweet texts tokenized

    Usage:
    python hydrate_tweet.py -i ../path/to/input/file.json 

    """
    def __init__(self, inpath):
        """
        """
        self.cfg_tw_api = import_config('twitter_api')
        self.inpath = inpath

    def connect_twitter_api(self):
        """
        Connect to Twitter API
        """
        self.api = twitter.Api(consumer_key=self.cfg_tw_api[
                                                    'consumer_key'],
                               consumer_secret=self.cfg_tw_api[
                                                    'consumer_secret'],
                               access_token_key=self.cfg_tw_api[
                                                    'access_token'],
                               access_token_secret=self.cfg_tw_api[
                                                    'access_secret'],
                               cache=None,
                               tweet_mode='extended')

    def hydrate_tweet(self, tweet_id):
        """
        Hydrate Tweet from ID. Due to rate limit, when it is reached the
        maximum number of request allowed it sleep for 5 minutes.
        More information about the Twitter Policy of rate limiting
        https://developer.twitter.com/en/docs/basics/rate-limiting.html
        """
        tweet_status = None
        try:
            tweet_status = self.api.GetStatus(tweet_id)
        except TwitterError, err:
            logging.error("Problem hydrating tweet ID %s: %s" % (
                                            tweet_id, err))

            if err.args[0][0]['code'] == 88:
                logging.error('Waiting 5 minutes and then retrying...')
                time.sleep(300)
                self.hydrate_tweet(tweet_id)

        return tweet_status

    def run(self, tweets):
        """
        Given as input a list of tweets_id and related entities annotated,
        hydrate each tweet using the id and associate the entities information.
        """
        self.connect_twitter_api()

        with io.open(self.cfg_tw_api['outfile_ent'] % self.inpath, 'w+',
                     newline='', encoding='utf-8') as csvfile,\
             io.open(self.cfg_tw_api['outfile_info'] % self.inpath, 'w+',
                     newline='', encoding='utf-8') as csvfile2,\
             io.open(self.cfg_tw_api['outfile_text'] % self.inpath, 'w+',
                     newline='', encoding='utf-8') as csvfile3:

            # Outfile Entities Check
            _writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            # Outfile Summary
            _writer2 = csv.writer(csvfile2, quoting=csv.QUOTE_ALL)
            # Write Headers
            _writer.writerow(['TWEET_ID', 'ENT', 'I', 'E', 'IOB_TAG', 'TYPE'])
            _writer2.writerow(['TWEET_ID', 'DATE', 'TEXT', 'ENT'])

            for tweet in tqdm(tweets):
                # Retrieve tweet info when possible, else skip
                tweet_status = self.hydrate_tweet(tweet['tweet_id'])
                if not tweet_status:
                    continue

                # Replace line breaks in tweet text
                tweet_text = tweet_status.full_text.replace('\n', ' ')
                # Remove URLs from text
                tweet_text = " ".join(filter(
                                      lambda x: x[0:4] != 'http',
                                      tweet_text.split()))

                # Summary file row
                row = [tweet['tweet_id'], tweet_status.created_at, tweet_text
                       ] + tweet['entities']

                try:
                    _writer2.writerow(row)
                except Exception, ex:
                    logging.error(ex)

                # Write out tweet text tokenized
                try:
                    csvfile3.write(' '.join(
                        [x for x in twk.tokenize(tweet_text)])+"\n")
                except Exception, ex:
                    logging.error(ex)

                # Check Entity annotations
                if tweet['entities']:
                    for entity in tweet['entities']:
                        try:
                            i, e, t = entity.split(',')
                        except ValueError, er:
                            print("Problem with tweet ID"
                                  "%d" % tweet['tweet_id'])
                            print(er)
                            sys.exit()

                        i, e = int(i), int(e)

                        # Check Entity types
                        if t not in ['Contributor', 'Work']:
                            logging.error('Entity not allowed: %s' % t)
                            continue
                        entity = tweet_status.full_text[i:e]

                        # Add IOB tags
                        start = True
                        for token in entity.split():
                            if start:
                                iob_tag = 'B'
                                start = False
                            else:
                                iob_tag = 'I'
                            ent_token = tweet_status.full_text[i:i+len(token)]

                            # Write out Entity annotated
                            row = [unicode(tweet['tweet_id']),
                                   ent_token, i, i + len(token),
                                   iob_tag, t]
                            try:
                                _writer.writerow(row)
                            except Exception, ex:
                                logging.error(ex)

                            # Update index
                            i = i + len(token) + 1


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Input file path")
    parser.add_argument("-l", "--logfile", type=str,
                        help="Log file path")
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg_parser()

    set_log_config(args.logfile, logging.ERROR)

    if not args.input:
        logging.error("Please insert options -i (input annotation "
                      "file path)")
        sys.exit()
    if not os.path.isfile(args.input):
        logging.error('Input file not valid')
        sys.exit()

    inpath, intype = args.input.rsplit('.', 1)

    if intype == 'json':
        with open(args.input, 'r') as jinf:
            tweets_annotated = json.load(jinf)
    else:
        logging.error("Input format not supported")
        sys.exit()

    if tweets_annotated:
        ha = HydrateAnnotated(inpath)
        ha.run(tweets_annotated)
    else:
        logging.error("No tweets annotated found")
