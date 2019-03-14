import os
import sys
import xml.etree.ElementTree as ET
import pickle
import stanfordcorenlp


class Word(object):
    def __init__(self, word, begin, end, sent, index):
        self.word = word
        self.begin = begin
        self.end = end
        self.sent = sent
        self.index = index


class Sentence(object):
    def __init__(self, begin, end, index):
        self.begin = begin
        self.end = end
        self.index = index
        self.words = []

    def get_text(self):
        return ' '.join([word.word for word in self.words])

    def sub_string(self, begin, end):
        return ' '.join([word.word for word in self.words[begin:end]])

    def get_original_string(self):
        result = ''
        offset = self.words[0].begin
        for word in self.words:
            for i in range(word.begin - offset - 1):
                result += ' '
            result += word.word
            offset = word.end
        return result

    @staticmethod
    def get_original_doc(sents):
        doc = ''
        offset = 1
        for sent in sents:
            for i in range(sent.begin - offset - 1):
                doc += '\n'
            doc += sent.get_original_string()
            offset = sent.end
        return doc

    def retokenize(self, nlp):
        # print(self.get_original_string())
        # print([word.word for word in self.words])
        nlp_token = nlp.word_tokenize(self.get_text().encode('UTF-8'))
        # print(nlp_token)
        i, j = 0, 0
        new_words = []
        while i < len(self.words) and j < len(nlp_token):
            word_i = self.words[i].word
            word_j = nlp_token[j]
            if word_j == '-LRB-':
                word_j = '('
            elif word_j == '-RRB-':
                word_j = ')'
            elif word_j == '-LSB-':
                word_j = '['
            elif word_j == '-RSB-':
                word_j = ']'
            
            if word_i == word_j:
                new_words.append(self.words[i])
                i += 1
                j += 1
                continue
            elif word_i.startswith(word_j):
                new_words.append(Word(word_j, self.words[i].begin, self.words[i].begin + len(word_j) - 1, self, 0))
                self.words[i] = Word(word_i[len(word_j):], self.words[i].begin + len(word_j), self.words[i].end, self, 0)
                j += 1
                continue
            elif word_j.startswith(word_i):
                k = i + 1
                expanded_word = word_i + self.words[k].word
                while not expanded_word.startswith(word_j):
                    k += 1
                    expanded_word += self.words[k].word
                if expanded_word == word_j:
                    new_words.append(Word(expanded_word, self.words[i].begin, self.words[k].end, self, 0))
                    i = k + 1
                    j += 1
                    continue
                else:
                    tail = len(expanded_word) - len(word_j)
                    new_words.append(Word(expanded_word, self.words[i].begin, self.words[k].end - tail, self, 0))
                    self.words[k] = Word(expanded_word[len(word_j):], self.words[k].end - tail + 1, self.words[k].end, self, 0)
                    i = k
                    j += 1
                    continue
            else:
                # skip and try to recover
                new_words.append(self.words[i])
                i += 1
                j += 1
                if i >= len(self.words):
                    break
                word_i = self.words[i].word
                word_j = nlp_token[j]
                if word_i.startswith(word_j) or word_j.startswith(word_i):
                    continue
                j += 1
                word_j = nlp_token[j]
                if word_i.startswith(word_j) or word_j.startswith(word_i):
                    continue
                print('Error: fail to recover')
                # raise Exception
                return

        for wid, word in enumerate(new_words):
            word.index = wid
        self.words = new_words


def read_ltf_offset(fname, out_fname=None, nlp=None):
    tree = ET.parse(fname)
    root = tree.getroot()
    flag = False
    sents = []
    for sent_id, seg in enumerate(root[0][0]):
        text = seg.find('ORIGINAL_TEXT').text
        sent = Sentence(int(seg.attrib['start_char']), int(seg.attrib['end_char']), sent_id)
        tokens = seg.findall('TOKEN')
        for tok_id, token in enumerate(tokens):
            sent.words.append(Word(token.text, int(token.attrib['start_char']), int(token.attrib['end_char']), sent, tok_id))
        if nlp:
            sent.retokenize(nlp)
        sents.append(sent)
    doc = Sentence.get_original_doc(sents)

    if out_fname:
        with open(out_fname + '.dump', 'wb') as f:
            pickle.dump(sents, f)
        with open(out_fname + '.txt', 'w') as f:
            f.write(doc)
    
    return sents, doc