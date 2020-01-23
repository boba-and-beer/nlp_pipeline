"""
This module aims to summarise text found in the documents.
"""
import nltk
import heapq

def nltk_summarise(text, stopwords_lang='english', max_sent_len=30,
    n_largest=7):
    """
    Apply NTLK summarisation. Uses word popularity. 
    Very simple summarisation algorithm. Does not reduce the number of words 
    by much.
    Example:
    >>> text = train['answer'][2]
    >>> print("===Summary===")
    >>> print(ntlk_summarise(text))
    >>> print("")
    >>> print("===Raw===")
    >>> print(text)
    """
    sentence_list = nltk.sent_tokenize(text)
    # Apply preprocessing here if required preprocessing her if required
    # Use stopwords
    stopwords = nltk.corpus.stopwords.words(stopwords_lang)

    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    # Calculate the word frequencies
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < max_sent_len:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
                    
    summary_sentences = heapq.nlargest(n_largest, sentence_scores, key=sentence_scores.get)
    summary = ''.join(summary_sentences)
    return summary    