import xml.etree.ElementTree as ET
from os.path import basename
from document import *
import sys 
import codecs

import os
from langdetect import detect

'''
for file in os.listdir("data/neg/ltf"):
    if file.endswith(".xml"):
        abs_file = os.path.join("data/neg/ltf", file)
        sents, doc = read_ltf_offset(abs_file)
        with codecs.open("data/neg/txt/" + basename(abs_file) + '.txt', 'w', encoding='utf-8') as f:
        	for sent in sents:
        		sentence = [word.word for word in sent.words]
        		f.write(' '.join(sentence) + '\n')

for file in os.listdir("data/neg/txt"):
	file_content = ""
	abs_file = os.path.join("data/neg/txt", file)
	with codecs.open(abs_file, "r", encoding='utf-8', errors='ignore') as f: 
		for line in f:
			file_content += line
	lang =  detect(file_content)
	if lang == 'en':
		os.rename(abs_file, os.path.join("data/neg/en",file))
	elif lang == 'uk':
		os.rename(abs_file, os.path.join("data/neg/uk",file))
	elif lang == 'ru':
		os.rename(abs_file, os.path.join("data/neg/ru",file))

	#print(detector.language)
'''
files_one = []
for file in os.listdir("data/neg/uk"):
	files_one.append(file)
files_two = []
for file in os.listdir("data/pos/uk"):
	print file
	if file in files_one:
		print file