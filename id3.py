import math
import sys
from fractions import Fraction
import numpy as np
import collections
from collections import Counter, defaultdict
import time


data_dict=[
	{'A1':'A', 'A2':70, 'A3':True, 'Class':'Class1'},
	{'A1':'A', 'A2':90, 'A3':True, 'Class':'Class2'},
	{'A1':'A', 'A2':85, 'A3':False, 'Class':'Class2'},
	{'A1':'A', 'A2':95, 'A3':False, 'Class':'Class2'},
	{'A1':'A', 'A2':70, 'A3':False, 'Class':'Class1'},
	{'A1':'B', 'A2':90, 'A3':True, 'Class':'Class1'},
	{'A1':'B', 'A2':78, 'A3':False, 'Class':'Class1'},
	{'A1':'B', 'A2':65, 'A3':True, 'Class':'Class1'},
	{'A1':'B', 'A2':75, 'A3':False, 'Class':'Class1'},
	{'A1':'C', 'A2':80, 'A3':True, 'Class':'Class2'},
	{'A1':'C', 'A2':70, 'A3':True, 'Class':'Class2'},
	{'A1':'C', 'A2':80, 'A3':False, 'Class':'Class1'},
	{'A1':'C', 'A2':80, 'A3':False, 'Class':'Class1'},
	{'A1':'C', 'A2':96, 'A3':False, 'Class':'Class1'}
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
-data_dict: data set that you would like to analyze
-attribute: the attribute you would like to get probabilities for
-attr_probabilities: the probabilities for the different values of
given attribute
-probabilities:  an array of the probabilities of the classifications
======================================================================
"""
def calcGain(data_dict,attribute, attr_probabilities,probabilities):
	H=calcH(probabilities)
	HAttribute=0.0
	denominator=len(data_dict)
	numerators=Counter(data[attribute] for data in data_dict)
	for attr, attr_ps in attr_probabilities.items():
		HT=calcH(attr_ps)
		HAttribute=HAttribute + Fraction(numerators[attr],denominator)*HT

	print("HAttribute: {0}".format(HAttribute))

	gain=H-HAttribute
	print("gain: {0}".format(gain))
	return gain

"""
===================================================================
getAttrProbabilities: gets the probabilities for a given attribute
-------------------------------------------------------------------
-data_dict: data set that you would like to analyze
-attribute: the attribute you would like to get probabilities for
-classification: the class that you are using to get resulting tree
===================================================================
"""
def getAttrProbabilities(data_dict, attribute, classification):
	probabilities = defaultdict(list)

	attr_denominators=Counter(data[attribute] for data in data_dict)
	class_counts = Counter((data[attribute], data[classification]) for data in data_dict)
	for data_pair, class_count in class_counts.items():
		attr, classification = data_pair
		probabilities[attr].append(Fraction(class_count, attr_denominators[attr]))
	return probabilities


"""
==============================================================
getClassProbabilities: calculates probabilities for H
--------------------------------------------------------------
-data_dict: data dictionary to look at
-classification: attribute that is used for the classification
===============================================================
"""
def getClassProbabilities(data_dict,classification):
	denominator=len(data_dict)
	classifications=[]
	sorted_dict=sorted(data_dict, key=lambda attribute: attribute[classification])
	current_class=sorted_dict[0][classification]
	class_count=0

	for i in sorted_dict:
		if(i[classification]==current_class):
			class_count+=1
		else:
			classifications.append(class_count)
			class_count=1
			current_class=i[classification]

	classifications.append(class_count)

	probabilities = [ Fraction(j,denominator) for j in classifications ]
	return probabilities

def partitionNumericalData(data_dict, attr, classification):
	# print("ATTR: ",attr)
	# median = sorted(data_dict, key=lambda d: -int(d[attr]))[len(data_dict) // 2][attr]
	#print(median)
	# smaller = [d for d in data_dict if int(d[attr]) <= int(median)]
	# bigger = [d for d in data_dict if int(d[attr]) > int(median)]
	# print(median)
	# print(smaller)
	# print(bigger)

	sort = sorted(data_dict, key=lambda d: -d[attr])
	median = sort[len(data_dict) // 2][attr]
	smaller = sort[len(data_dict) // 2:] # [d for d in data_dict if d[attr] <= median]
	bigger = sort[:len(data_dict) // 2] # [d for d in data_dict if d[attr] > median]
	return smaller, bigger, median

"""
==============================================================
getSplittingAttribute: gets the splitting attribute of the
given data set
--------------------------------------------------------------
-data_dict: data dictionary to look at
-attr_types: data dictionary that has the type of each attribute
in the given data set
===============================================================
"""
def getSplittingAttribute(data_dict,attr_types):
	attr_gains={}
	classification=None

	for attr_class in attr_types:
		if(attr_class[1]=="class"):
			classification=attr_class[0]


	probabilities=getClassProbabilities(data_dict, classification)
	for attr, attr_type in attr_types:
		if attr_type == "categorical":
			print("ATTRIBUTE: {0}".format(attr))
			print("===========================")
			attr_probabilities=getAttrProbabilities(data_dict, attr, classification)
			attr_gains[attr]=calcGain(data_dict, attr, attr_probabilities, probabilities)
			print("\n")
		elif attr_type == "numerical":
			print("ATTRIBUTE: {0}".format(attr))
			print ("===========================")
			smaller, bigger, median = partitionNumericalData(data_dict, attr, classification)

			H_smaller = calcH(getClassProbabilities(smaller, classification))
			H_bigger = calcH(getClassProbabilities(bigger, classification))
			HTAttribute = Fraction(len(smaller),len(data_dict))*H_smaller + Fraction(len(bigger),len(data_dict))*H_bigger
			H = calcH(probabilities)
			attr_gains[attr] = H - HTAttribute
			print ("\n")
		else:
			print("{} ATTRIBUTE: {}".format(attr_type, attr))

	non_class_attr_types = [(a, a_type) for a, a_type in attr_types if a_type != 'class']
	split_attr, split_attr_type = non_class_attr_types[0]
	for attr, attr_type in non_class_attr_types:
		if(attr_gains[attr]>attr_gains[split_attr]):
			split_attr, split_attr_type = attr, attr_type

	print("Splitting Attribute = {0}".format(split_attr))
	return split_attr, split_attr_type

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
def getDataDict(attr_types,filePath):
	data_dict=[]
	with open(filePath, 'r') as f:
		for line in f:
			line = line.strip()
			if '?' in line:
				continue

			data_set_row=line.split(",")
			data_dict_row={}
			index=0
			for attr, attr_type in attr_types:
				if attr_type == 'numerical':
					data_dict_row[attr] = float(data_set_row[index].strip())
				else:
					data_dict_row[attr] = data_set_row[index].strip()
				index+=1
			data_dict.append(data_dict_row)
	return data_dict

node_count = 0
def createDecsionTree(data_dict, attr_types, parent_name, branch_name):
	global node_count
	node_count+=1
	classification=None
	new_dicts={}

	for attr_class in attr_types:
		if attr_class[1] == "class":
			classification=attr_class[0]

	#print("remaining data", data_dict[0][classification])
	active_classes = { datum[classification] for datum in data_dict }
	print("active classes", active_classes)
	if len(active_classes) > 1:
		split_attr, split_attr_type = getSplittingAttribute(data_dict, attr_types)

		node_name = split_attr + '_' + str(node_count)
		print('"{}" [label="{}"];'.format(node_name, split_attr), file=sys.stderr)

		if parent_name is not None:
			print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name, branch_name), file=sys.stderr)

		if split_attr_type == 'categorical':
			sorted_dict=sorted(data_dict, key=lambda a: (a[split_attr], a[classification]))
			split_attr_nodes=list(Counter(data[split_attr] for data in data_dict).keys())
			new_dicts={ node: [a for a in sorted_dict if a[split_attr] == node]
						for node in split_attr_nodes }

			for node in new_dicts:
				attr_dict=new_dicts[node]
				for d in attr_dict:
					d.pop(split_attr, None)

			attr_types = [a for a in attr_types if a[0] != split_attr]

			for node in new_dicts:
				attr_dict=new_dicts[node]
				createDecsionTree(attr_dict,attr_types, node_name,node)
		else:
			assert split_attr_type == 'numerical', split_attr_type
			smaller, bigger, median = partitionNumericalData(data_dict, split_attr, classification)
			createDecsionTree(smaller, attr_types, node_name, "<= " + str(median))
			createDecsionTree(bigger, attr_types, node_name, "> " + str(median))

	else:
		class_name = data_dict[0][classification]
		node_name = class_name+'_'+str(node_count)
		print('"{}" [label="{}"];'.format(node_name, class_name), file=sys.stderr)
		print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name,branch_name), file=sys.stderr)
		return

"""
==============================
			MAIN
==============================
"""
def main():
	start=time.time()
	# attr_types=getAttrTypes("./mnist.attr")
	# data_dict=getDataDict(attr_types,"./mnist.data")
	attr_types=getAttrTypes("./attributes.txt")
	data_dict=getDataDict(attr_types,"./dataset_large.txt")
	#attr_types=[["A1","categorical"],["A2","numerical"],["A3","categorical"],["Class","class"]]
	#getSplittingAttribute(data_dict,attr_types)
	print("digraph g{", file=sys.stderr)
	createDecsionTree(data_dict,attr_types,None,None)
	print("}", file=sys.stderr)

	print(time.time()-start)




if __name__ == '__main__':
	main()
