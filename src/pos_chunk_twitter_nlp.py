#!/usr/bin/env python
# coding: utf-8

import re

import twitter_nlp.python.pos_tagger_stdin as pstag
import twitter_nlp.python.chunk_tagger_stdin as chtag
import twitter_nlp.python.twokenize as twk


class PosChunkTagger(object):
    """
    Wrapper of twitter_nlp package. For more information about the package
    https://github.com/aritter/twitter_nlp
    """
    def __init__(self):
        """
        """
        self.posTagger = pstag.PosTagger()
        self.chunkTagger = chtag.ChunkTagger()

    def tag_sentence(self, sentence):
        """
        Given as input a sentence, return sentence with POS and Chunk tags
        """
        words = twk.tokenize(sentence)

        pos = self.posTagger.TagSentence(words)
        pos = [re.sub(r':[^:]*$', '', p) for p in pos]
        word_pos = zip(words, [p.split(':')[0] for p in pos])

        chunk = self.chunkTagger.TagSentence(word_pos)
        chunk = [c.split(':')[0] for c in chunk]

        output = [(words[x], pos[x], chunk[x]) for x in range(len(words))]

        return output

if __name__ == '__main__':

    inp = 'Just for testing a random input'
    tagger = PosChunkTagger()
    output = tagger.tag_sentence(inp)
    print "Input sentence: \n\t'%s'" % inp
    print "Output sentence with tags: \n\t%s" % output
