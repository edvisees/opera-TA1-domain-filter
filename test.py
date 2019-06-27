#from sklearn.datasets import fetch_20newsgroups
import os
import codecs
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
import pickle as pkl
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from langdetect import detect
from keras.models import load_model
import argparse
import xml.etree.ElementTree as ET
import sys
import traceback

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


def filter(content, tfidf, filter):
	
	tfidf_feature = tfidf.transform(content)
	#predict = filter.predict_classes(tfidf_feature)
	predict = filter.predict(tfidf_feature)
	predict = predict[:,1]
	#print predict.shape
	predict[predict >=0.1] = 1
	predict[predict < 0.1] = 0
	return predict

def main(data, tfidf, filter_model):
	domain = filter(data, tfidf, filter_model)
	return domain
def load_models(lang):
	root_path = './resources/xiang/domain_filter'
	tfidf_path = 'tfidf0{}.pkl'.format(lang)
	filter_path = '{}_best.hdf5'.format(lang)
	with open(os.path.join(root_path, tfidf_path)) as f:
		tfidf = pkl.load(f)
	filter_model = load_model(os.path.join(root_path, filter_path))
	return tfidf, filter_model

def select_files(files, predict):
	select_files = [files[i] for i, x in enumerate(predict) if x == 0]
	return select_files

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	#parser.add_argument('file_name', type=str, help='file name')
	parser.add_argument('folder_name', type=str, help='folder name')
	args = parser.parse_args()
	#file_name = args.file_name
	folder_name = args.folder_name
	batch_size = 128
	en_tfidf, en_filter = load_models('en')
	ru_tfidf, ru_filter = load_models('ru')
	uk_tfidf, uk_filter = load_models('uk')
	en_files = []
	ru_files = []
	uk_files = []
	en_files_name = []
	ru_files_name = []
	uk_files_name = []
	en_domain = []
	ru_domain = []
	uk_domain = []
	in_domain_files = []
	for file_name in os.listdir(folder_name):
		#with codecs.open(os.path.join(folder_name, file_name), encoding='utf-8') as f:
		if not file_name.endswith(".xml"):
			continue
		try:
			tree = ET.parse(os.path.join(folder_name, file_name))
		except:
			sys.stderr.write("ERROR: Exception occured while processing " + file_name)
			traceback.print_exc()
			continue
		root = tree.getroot()
		flag = False
		sents = []
		content = ""
		for sent_id, seg in enumerate(root[0][0]):
			text = seg.find('ORIGINAL_TEXT').text
			content = "".join((content, text))
			#print content
			#print file_name
		try:
			lang =  detect(content)
		except:
			continue
		if lang == 'en':
			en_files.append(content)
			en_files_name.append(os.path.join(folder_name, file_name))
			if len(en_files) == batch_size:
				res = main(en_files, en_tfidf, en_filter)

				en_domain.extend(res)
				in_domain_files.extend(select_files(en_files_name, res))
				del en_files[:]
				del en_files_name[:]
		elif lang == 'ru':
			ru_files_name.append(os.path.join(folder_name, file_name))
			ru_files.append(content)
			if len(ru_files) == batch_size:
				res = main(ru_files, ru_tfidf, ru_filter)
				ru_domain.extend(res)
				in_domain_files.extend(select_files(ru_files_name, res))
				del ru_files[:]
				del ru_files_name[:]
		elif lang == 'uk':
			uk_files.append(content)
			uk_files_name.append(os.path.join(folder_name, file_name))
			if len(uk_files) == batch_size:
				res = main(uk_files, uk_tfidf, uk_filter)
				uk_domain.extend(res)
				in_domain_files.extend(select_files(uk_files_name, res))
				del uk_files[:]
				del uk_files_name[:]
	if len(en_files) > 0:
		res = main(en_files, en_tfidf, en_filter)
		en_domain.extend(res)
		in_domain_files.extend(select_files(en_files_name, res))

	if len(ru_files) > 0:
		res = main(ru_files, ru_tfidf, ru_filter)
		ru_domain.extend(res)
		in_domain_files.extend(select_files(ru_files_name, res))
	if len(uk_files) > 0:
		res = main(uk_files, uk_tfidf, uk_filter)
		uk_domain.extend(res)
		in_domain_files.extend(select_files(uk_files_name, res))
	#print len(en_domain), len(ru_domain), len(uk_domain)
	#print float(sum(en_domain)) / len(en_domain)
	#print float(sum(ru_domain)) / len(ru_domain)
	#print float(sum(uk_domain)) / len(uk_domain)
	#print len(in_domain_files)
	parent_folder = os.path.abspath(os.path.join(folder_name, '..'))
	out_domain_folder = os.path.join(parent_folder, 'out_domain')
	check_folder = os.path.exists(out_domain_folder)
	if not check_folder:
		os.makedirs(out_domain_folder)
	for files in in_domain_files:
		os.rename(files, os.path.join(out_domain_folder, os.path.split(files)[-1]))
	#main(file_name)






