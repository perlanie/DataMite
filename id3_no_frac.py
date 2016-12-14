import math
import sys
import numpy as np
import collections
from collections import Counter, defaultdict
import time 

data_dict = [
	['A', 70, True, 'Class1'],
	['A', 90, True, 'Class2'],
	['A', 85, False, 'Class2'],
	['A', 95, False, 'Class2'],
	['A', 70, False, 'Class1'],
	['B', 90, True, 'Class1'],
	['B', 78, False, 'Class1'],
	['B', 65, True, 'Class1'],
	['B', 75, False, 'Class1'],
	['C', 80, True, 'Class2'],
	['C', 70, True, 'Class2'],
	['C', 80, False, 'Class1'],
	['C', 80, False, 'Class1'],
	['C', 96, False, 'Class1']
]


"""
========================================
calcH: calculates the H for the Entropy
----------------------------------------
-probabilities: an array of the
probabilities of the classifications
========================================
"""
def calcH(probabilities):
	return sum(-p*math.log(p,2) for p in probabilities)


"""
=====================================================================
calcGain: calculates gain of each attributes
---------------------------------------------------------------------
-data: data set that you would like to analyze
-attr_idx: attribute index
-attr_probabilities: the probabilities for the different values of
given attribute
-probabilities:  an array of the probabilities of the classifications
======================================================================
"""
def calcGain(data, attr_idx, attr_probabilities, probabilities):
	H=calcH(probabilities)
	HAttribute=0.0
	denominator=len(data)
	numerators=Counter(d[attr_idx] for d in data)
	for attr, attr_ps in attr_probabilities.items():
		HT=calcH(attr_ps)
		HAttribute=HAttribute + (numerators[attr] / denominator)*HT

	# print("HAttribute: {0}".format(HAttribute))

	gain=H-HAttribute
	# print("gain: {0}".format(gain))
	return gain

"""
===================================================================
getAttrProbabilities: gets the probabilities for a given attribute
-------------------------------------------------------------------
-data: data set that you would like to analyze
-attr_idx: the attribute index you would like to get probabilities for
-class_idx: the class index that you are using to get resulting tree
===================================================================
"""
def getAttrProbabilities(data, attr_idx, class_idx):
	probabilities = defaultdict(list)

	attr_denominators=Counter(d[attr_idx] for d in data)
	class_counts = Counter((d[attr_idx], d[class_idx]) for d in data)
	for data_pair, class_count in class_counts.items():
		attr, classification = data_pair
		probabilities[attr].append(class_count / attr_denominators[attr])
	return probabilities


"""
==============================================================
getClassProbabilities: calculates probabilities for H
--------------------------------------------------------------
-data: data dictionary to look at
-class_idx: the class index that you are using to get resulting tree
===============================================================
"""
def getClassProbabilities(data, class_idx):
	denominator = len(data)
	class_counts = Counter(d[class_idx] for d in data)
	probabilities = [ count / denominator for cls, count in class_counts.items() ]
	return probabilities

"""
==============================================================
partitionNumericalData: partitions numerical data
--------------------------------------------------------------
-data: data dictionary to look at
-attr_idx: attribute index
-class_idx: class index
===============================================================
"""
def partitionNumericalData(data, attr_idx, class_idx):
	avg = sum(d[attr_idx] for d in data) / len(data)
	smaller = []
	bigger = []
	for d in data:
		if d[attr_idx] <= avg:
			smaller.append(d)
		else:
			bigger.append(d)
	return smaller, bigger, avg


"""
==============================================================
getSplittingAttribute: gets the splitting attribute of the
given data set
--------------------------------------------------------------
-data: data dictionary to look at
-attr_types: data dictionary that has the type of each attribute
in the given data set
===============================================================
"""
def getSplittingAttribute(data, attr_types):
	attr_gains=[None] * len(attr_types)

	class_idx = None
	for i, attr_class in enumerate(attr_types):
		if attr_class[1] == "class":
			class_idx = i

	probabilities = getClassProbabilities(data, class_idx)
	for attr_idx, attr_info in enumerate(attr_types):
		attr, attr_type = attr_info
		if attr_type == "categorical":
			attr_probabilities = getAttrProbabilities(data, attr_idx, class_idx)
			attr_gains[attr_idx] = calcGain(data, attr_idx, attr_probabilities, probabilities)
		elif attr_type == "numerical":
			smaller, bigger, median = partitionNumericalData(data, attr_idx, class_idx)
			H_smaller = calcH(getClassProbabilities(smaller, class_idx))
			H_bigger = calcH(getClassProbabilities(bigger, class_idx))
			HTAttribute = (len(smaller) / len(data))*H_smaller + (len(bigger) / len(data))*H_bigger
			H = calcH(probabilities)
			attr_gains[attr_idx] = H - HTAttribute
		else:
			print("WEIRD: {} ATTRIBUTE: {}".format(attr_type, attr))

	max_attr_idx = None
	for i in range(len(attr_gains)):
		if attr_gains[i] is None:
			continue
		if max_attr_idx is None or attr_gains[i] > attr_gains[max_attr_idx]:
			max_attr_idx = i

	print("Splitting Attribute = {0}".format(attr_types[max_attr_idx]))
	return max_attr_idx


"""
==============================================================
getAttrTypes: gets the attrutes and their types
--------------------------------------------------------------
-filePath: the path of the file with the attributes and their
types
==============================================================
"""
def getAttrTypes(filePath):
	attr_types=[]
	with open(filePath, 'r') as fileHandler:
		for line in fileHandler:
			line = line.strip()
			attr=line.split(":")
			attr_types.append([a.strip() for a in attr])
	return attr_types


"""
==============================================================
getDataDict: reads in a data set and turns it into a data
dictionary
--------------------------------------------------------------
-attr_types: the attributes along with what types they are
-filePath: the path of the file with the data set
===============================================================
"""
def getDataDict(attr_types, filePath):
	data=[]
	attr_count=len(attr_types)
	with open(filePath, 'r') as f:
		for line in f:
			line = line.strip()
			if '?' in line:
				continue

			data_row = []
			for attr_info, elem in zip(attr_types, line.split(",")):
				attr_type = attr_info[1]
				if attr_type == 'numerical':
					data_row.append(float(elem.strip()))
				else:
					data_row.append(elem.strip())
			
			if attr_count==len(data_row):
				data.append(data_row)
	return data



"""
==============================================================
createDecsionTree: reads in a data set and turns it into a data
dictionary
--------------------------------------------------------------
-data: data dictionary to look at
-attr_types: the attributes along with what types they are
-parent_name: name of the parent node
-branch_name: name of the branch the parent expands on
===============================================================
"""
node_count = 0
def createDecsionTree(data, attr_types, parent_name, branch_name):
	global node_count
	node_count+=1

	class_idx = None
	for i, attr_info in enumerate(attr_types):
		if attr_info[1] == "class":
			class_idx = i

	active_classes = { datum[class_idx] for datum in data }
	if len(active_classes) > 1 and len(attr_types) > 1:
		split_attr_idx = getSplittingAttribute(data, attr_types)
		split_attr, split_attr_type = attr_types[split_attr_idx]

		node_name = split_attr + '_' + str(node_count)

		if split_attr_type == 'categorical':
			if parent_name is not None:
				print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name, branch_name), file=sys.stderr)
			print('"{}" [label="{}"];'.format(node_name, split_attr), file=sys.stderr)

			split_attr_nodes = Counter(d[split_attr_idx] for d in data).keys()

			new_dicts={}
			for node in split_attr_nodes:
				new_dicts[node]=[]

			for d in data:
				attr=d[split_attr_idx]
				del d[split_attr_idx]
				new_dicts[attr].append(d)


			attr_types = [a for i, a in enumerate(attr_types) if i != split_attr_idx]
			assert len(data[0]) == len(attr_types)
			#print(data[0])
			#print([t[1] for t in attr_types])
			for node in new_dicts:
				attr_data = new_dicts[node]
				createDecsionTree(attr_data, attr_types, node_name, node)
		else:
			assert split_attr_type == 'numerical', split_attr_type
			smaller, bigger, branching_num = partitionNumericalData(data, split_attr_idx, class_idx)
			if smaller == [] or bigger == []:
				print('We have competing evidence here')
			else:
				if parent_name is not None:
					print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name, branch_name), file=sys.stderr)
					print('"{}" [label="{}"];'.format(node_name, split_attr), file=sys.stderr)
				createDecsionTree(smaller, attr_types, node_name, "<= " + str(branching_num))
				createDecsionTree(bigger, attr_types, node_name, "> " + str(branching_num))

	else:
		class_name = data[0][class_idx]
		node_name = class_name + '_' + str(node_count)
		print('"{}" [label="{}"];'.format(node_name, class_name), file=sys.stderr)
		print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name,branch_name), file=sys.stderr)

"""
==============================
			MAIN
==============================
"""
def main():
	start=time.time()
	#attr_types=[["A1","categorical"],["A2","numerical"],["A3","categorical"],["Class","class"]]
	attr_types=getAttrTypes("./tic-tac-toe_attrs.txt")
	data_dict=getDataDict(attr_types,"./tic-tac-toe.txt")
	
	print("digraph g{", file=sys.stderr)
	createDecsionTree(data_dict,attr_types,None,None)
	print("}", file=sys.stderr)
	print(time.time()-start)




if __name__ == '__main__':
	main()