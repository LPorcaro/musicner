#!/usr/bin/env python
# encoding: utf-8

import io
import argparse
import logging
import string
import twitter_nlp.python.twokenize as twk

from backports import csv
from dateutil.parser import parse
from Levenshtein import jaro_winkler

from utils import import_config, set_log_config


class ScheduleMatcher(object):
    """
    It searches for matches between the entities annotated in the schedule
    and the user-generated tweets. It writes the results in a text file
    in CoNLL format. The input parameters are the input file paths and
    the thresholds:
    time_tsl: time-distance between schedule tweet and user-generated tweet
    work_tsl: string similarity for Work entities
    contr_tsl: string similarity for Contributor entities

    Usage:
    python schedule_matcher.py
    -w work_tsl (float)
    -c contr_tsl (float)
    -t time_tsl (int)
    -i ../path/to/UGC_INPUTFILE_summary.csv
    -s ../path/to/SCHEDULE_INPUTFILE_summary.csv

    """
    def __init__(self, input_file, schedule_file, limit, work_tsl,
                 contr_tsl, time_tsl):
        """
        """
        self.cfg_match = import_config('matcher')

        self.input_file = input_file
        self.schedule_file = schedule_file
        self.limit = limit
        self.work_tsl = work_tsl
        self.contr_tsl = contr_tsl
        self.time_tsl = time_tsl

        self.DictTweets = self.import_ugc_tweets()
        self.DictSched = self.import_schedule()
        self.stopwords = self.import_stopwords()

        # Initialize counters
        (self.tp_count, self.fp_count, self.tn_count, self.fn_count,
         self.tp_c_count, self.fp_c_count, self.tn_c_count, self.fn_c_count
         ) = (0,)*8

        self.outfile = "../results/schedule_matcher_%s_%s_%s.txt" % (
                            work_tsl, contr_tsl, time_tsl)

    def import_stopwords(self):
        """
        Import in a set the stopwords defined in the file defined in the
        config
        """
        logging.info('Importing stopwords...')
        stopwords = set()
        with io.open(self.cfg_match['stopwords'], newline='',
                     encoding='utf-8') as csvfile:
            _reader = csv.reader(csvfile)
            for row in _reader:
                stopwords.add(row[0].lower())

        logging.info("Done!")
        return stopwords

    def import_ugc_tweets(self):
        """
        Import in a Dict the UGC tweets information, using as key
        the tweet ID
        """
        logging.info('Importing User-Generated tweets...')
        DictTweets = {}
        with io.open(self.input_file, newline='', encoding='utf-8') as infile:
            _reader = csv.reader(infile)
            next(_reader)
            for row in _reader:
                tweet_id, created_at, text = row[0:3]
                DictTweets[tweet_id] = {}
                DictTweets[tweet_id]['created_at'] = created_at
                DictTweets[tweet_id]['text'] = text
                DictTweets[tweet_id]['entities'] = []
                if len(row) > 3:
                    DictTweets[tweet_id]['entities'] = row[3:]

        logging.info("Done!")
        return DictTweets

    def import_schedule(self):
        """
        Import in a Dict the schedule tweets information, using as key
        the tweet ID
        """
        count = 0
        DictSched = {}
        with io.open(self.schedule_file, newline='', encoding='utf-8') as inf:
            _reader = csv.reader(inf)
            next(_reader)
            # Iterate over Schedule Tweets
            for row in _reader:
                sch_tweet_id, created_at, text = row[:3]
                if len(row) > 3:
                    entities = row[3:]

                if self.limit and count == self.limit:
                    break
                if count == 0:
                    logging.info("Processing Schedule Tweets...")
                elif count % 1000 == 0:
                    logging.info("Processed %d Schedule Tweets", count)
                count += 1

                DictSched[sch_tweet_id] = {}
                DictSched[sch_tweet_id]['text'] = text
                DictSched[sch_tweet_id]['entities'] = (
                    self.extract_schedule_entities(text, entities))
                DictSched[sch_tweet_id]['created_at'] = parse(created_at)

        logging.info("Done!")
        return DictSched

    def extract_schedule_entities(self, text, entities):
        """
        Extract the entities annotated from the text of the schedule tweet
        """
        entities_token = set()
        for entity in entities:
            i, e, etype = entity.split(',')
            token = text[int(i):int(e)]
            entities_token.add((token, etype))
        return entities_token

    def search_schedule_matches(self, sch_tweet_id, tweet_text_tokens,
                                entities_matched, entities_token_matched,
                                matched_sch_ids):
        """
        Search for matches betweet schedule tweets annotated entities and
        ugc tweet. It returns the matches found.
        """
        # Iterate over the entities of the schedule
        for sch_entity in self.DictSched[sch_tweet_id]['entities']:
            sch_entity_strip = sch_entity[0].lower().split()
            sch_entity_type = sch_entity[1]
            # Sanity check over schedule entities annotated
            if not sch_entity_strip:
                continue

            # Get token matched between ugc and schedule tweet
            token_matches = [
                (t, sch_entity_type) for t in sch_entity_strip if [
                    s for s in tweet_text_tokens
                    if jaro_winkler(t.lower(), s.lower()) >= 0.95
                    ] and
                t not in self.stopwords and
                t not in string.punctuation]

            # Compute score for string similarity
            score = len(token_matches)/float(len(sch_entity_strip))

            # Check if string similarity conditions are valid
            if (sch_entity_type.endswith('Contributor') and
                    score >= self.contr_tsl) or \
               (sch_entity_type.endswith('Work') and
                    score >= self.work_tsl):
                # Discard matches againts already matched entities
                if sch_entity_strip not in entities_matched:
                    entities_matched.append(sch_entity_strip)
                    if token_matches not in entities_token_matched:
                        entities_token_matched += token_matches
                    matched_sch_ids.append(sch_tweet_id)

        return entities_matched, entities_token_matched, matched_sch_ids

    def convert_entity_iob(self, tweet_ent_split, iob, prev_type, token):
        """
        Convert entity to IOB format
        """
        etype = [x[1] for x in tweet_ent_split if x[0] == token][0]

        if not iob and etype != 'O':
            iob = 'B'
            prev_type = etype
        elif etype == 'O':
            prev_type = None
        elif etype == prev_type:
            iob = 'I'
        elif etype != prev_type and etype != 'O':
            iob = 'B'
            prev_type = etype

        ann_entity = '-'.join([iob, etype])

        return ann_entity, prev_type, iob

    def write_results(self, outf, tweet_text_tokens, tweet_ent_split,
                      entities_token_matched):
        """
        Write out results in CoNLL format.
        """
        iob = None
        prev_type = None
        ann_iob = None
        ann_prev_type = None

        # Get info about entities annotated
        for token in tweet_text_tokens:
            if token in [x[0] for x in tweet_ent_split]:
                ann_entity, ann_prev_type, ann_iob = self.convert_entity_iob(
                    tweet_ent_split, ann_iob, ann_prev_type, token)
            else:
                ann_entity = 'O'

            # Get info about entities predicted
            if token in [x[0] for x in entities_token_matched]:
                pred_entity, prev_type, iob = self.convert_entity_iob(
                    entities_token_matched, iob, prev_type, token)
            else:
                pred_entity = 'O'

            outf.write(
                unicode('%s %s %s\n' % (token, ann_entity, pred_entity)))
        outf.write(unicode('\n'))

    def run(self):
        """
        Iterate over the UG tweets looking for matches with
        the schedule.
        """
        with io.open(self.outfile, 'w+', newline='', encoding='utf-8') as outf:
            logging.info("Looking for matches...")
            for tweet_id in self.DictTweets:
                tweet_date = parse(self.DictTweets[tweet_id]['created_at'])
                tweet_text_tokens = twk.tokenize(
                    self.DictTweets[tweet_id]['text'].lower())

                # Get UG Tweet Entities text splitted
                tweet_ent_split = []
                for tweet_entity in self.DictTweets[tweet_id]['entities']:
                    s, e, t = tweet_entity.split(',')
                    s, e = int(s), int(e)
                    tweet_ent_split += [
                        (x, t) for x in
                        self.DictTweets[tweet_id]['text'][s:e].lower().split()]

                # Iterate over Schedule Tracks looking for the best match
                entities_matched = []
                entities_token_matched = []
                matched_sch_ids = []
                for sch_tweet_id in self.DictSched:
                    sch_date = self.DictSched[sch_tweet_id]['created_at']
                    diff = (sch_date - tweet_date).total_seconds()

                    # If the time distance between schedule and ug tweet
                    # is lower than the threshold, search for matches
                    if abs(diff) < self.time_tsl:
                        (entities_matched, entities_token_matched,
                            matched_sch_ids) = self.search_schedule_matches(
                                        sch_tweet_id,
                                        tweet_text_tokens,
                                        entities_matched,
                                        entities_token_matched,
                                        matched_sch_ids)

                # Check matches in debug mode
                if entities_token_matched:
                    logging.debug("New match found!")
                    logging.debug("Tokens matched: <%s>", ', '.join(
                                    [x[0] for x in entities_token_matched]))
                    for m in set(matched_sch_ids):
                        logging.debug("Track matched (%s): '%s'",
                                      self.DictSched[m]['created_at'].strftime(
                                            '%Y-%m-%d %H:%M:%S'),
                                      self.DictSched[m]['text'])

                    logging.debug("Original tweet (%s):, '%s'",
                                  parse(self.DictTweets[tweet_id]['created_at']
                                        ).strftime('%Y-%m-%d %H:%M:%S'),
                                  self.DictTweets[tweet_id]['text'])

                    logging.debug("Tweet Entities annotated: <%s> ", ', '.join(
                        [x[0] for x in tweet_ent_split]))

                self.write_results(outf, tweet_text_tokens, tweet_ent_split,
                                   entities_token_matched)

        logging.info("Done!")


def arg_parser():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, dest='input_file',
                        help="Input file with UGC tweet")
    parser.add_argument("-s", "--schedule-file", type=str, dest='sched_file',
                        help="Input file schedule tracks")
    parser.add_argument("-l", "--logfile", type=str, help="Log file path")
    parser.add_argument("-L", "--limit", type=int, dest='limit',
                        help="Limit number of Schedule Tracks to process")
    parser.add_argument("-w", "--work-tsl", type=float, dest='work_tsl',
                        help="Work threshold for matching")
    parser.add_argument("-c", "--contr-tsl", type=float, dest='contr_tsl',
                        help="Contributor threshold for matching")
    parser.add_argument("-t", "--time-tsl", type=int, dest='time_tsl',
                        help="Time-distance threshold for matching")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = arg_parser()
    set_log_config(args.logfile, logging.INFO)
    sm = ScheduleMatcher(args.input_file,
                         args.sched_file,
                         args.limit,
                         args.work_tsl,
                         args.contr_tsl,
                         args.time_tsl)
    sm.run()
