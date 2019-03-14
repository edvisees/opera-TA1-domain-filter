import sys
input_file = sys.argv[1]
output_file = sys.argv[2]
'''
idx = 0
with open(input_file) as f, open(output_file, 'w') as o:
	for line in f:
		if idx % 7 == 0:
			o.write(line.split(':')[1])
		elif idx % 7 == 5:
			type_str = line.strip().split()
			type_list = []
			for t in type_str:
				#print t
				if 'ldcOnt' in t:
					type_list.append(t.split(':')[1])
			o.write(' '.join(type_list) + '\n')
		idx += 1
'''

rel_conversion = {}
with open('ner_map.txt') as f:
	for line in f:
		types = line.strip().split()
		rel_conversion[types[1]] = types[0].upper()
idx = 0
with open(input_file) as f, open(output_file, 'w') as o:
	for line in f:
		if idx % 2 == 0:
			o.write(line)
		else:
			type_list = line.split()
			for i in range(len(type_list)):
				t = type_list[i]
				#print t
				if t in rel_conversion:

					type_list[i] = rel_conversion[t]
			o.write(' '.join(type_list) + '\n')
		idx += 1
