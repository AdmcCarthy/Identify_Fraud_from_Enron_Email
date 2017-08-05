#!/usr/bin/python
"""
    parse out email txt
    ~~~~~~~~~~~~~~~~~~~~

    Module to parse through an email and
    extract text, can include a stemmer
    to group similar words together.
"""
from __future__ import print_function
from nltk.stem.snowball import SnowballStemmer
import string
import re

try:
    maketrans = ''.maketrans
except AttributeError:
    # fallback for Python 2
    from string import maketrans

def parseOutText(f, stem=True):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = (content[1].translate(string.maketrans("", ""),
                       string.punctuation))
        if stem:
            # Split the text string into individual words, stem each word,
            # and append the stemmed word to words (make sure there's a single
            # space between each stemmed word).
            words = []
            stemmer = SnowballStemmer("english")

            # REGEX to seperate out all words from a string into a list
            wordList = re.sub("[^\w]", " ",  text_string).split()

            for word in wordList:
                stem_word = stemmer.stem(word)
                words.append(str(stem_word))

            # ' ' implies the what should be
            # between the word joins.
            words = ' '.join(words)

        else:
            words = text_string
    return words


def main():
    ff = open("test_email.txt", "r")
    text = parseOutText(ff)
    print(text)


if __name__ == '__main__':
    main()
