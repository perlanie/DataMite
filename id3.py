import math
import sys
from fractions import Fraction
import numpy as np
import collections
from collections import Counter

data_dict=[
	{'A1':'A','A2':70,'A3':True, 'Class':'Class1'},
    {'A1':'A','A2':90,'A3':True, 'Class':'Class2'},
   	{'A1':'A','A2':85,'A3':False, 'Class':'Class2'},
   	{'A1':'A','A2':95,'A3':False, 'Class':'Class2'},
    {'A1':'A','A2':70,'A3':False, 'Class':'Class1'},
    {'A1':'B','A2':90,'A3':True, 'Class':'Class1'},
    {'A1':'B','A2':78,'A3':False, 'Class':'Class1'},
    {'A1':'B','A2':65,'A3':True, 'Class':'Class1'},
    {'A1':'B','A2':75,'A3':False, 'Class':'Class1'},
    {'A1':'C','A2':80,'A3':True, 'Class':'Class2'},
    {'A1':'C','A2':70,'A3':True, 'Class':'Class2'},
    {'A1':'C','A2':80,'A3':False, 'Class':'Class1'},
    {'A1':'C','A2':80,'A3':False, 'Class':'Class1'},
    {'A1':'C','A2':96,'A3':False, 'Class':'Class1'}   
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
	H = 0.0

	for p in probabilities:
		H=H-p*math.log(p,2)
		#print("H=H-{0}".format(p*math.log(p,2)))

	print("Entropy: {0}".format(H))
	return H


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
	prob_index=0
	denominator=len(data_dict)
	numerators=Counter(data[attribute] for data in data_dict)
	for attr in attr_probabilities:
		HT=0.0
		for ap in attr_probabilities[attr]:
			HT=HT-ap*math.log(ap,2)
		HAttribute=HAttribute + (Fraction(numerators[attr],denominator)*HT)
		prob_index+=1
			
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
def getAttrProbabilities(data_dict,attribute, classification):
	sorted_dict=sorted(data_dict, key=lambda a: (a[attribute],a[classification]))
	probabilities={}

	current_class=sorted_dict[0][classification]
	current_attr=sorted_dict[0][attribute]
	class_count=0
	attr_probabilities=[]
	attr_denominators=Counter(data[attribute] for data in data_dict)
	

	for a in sorted_dict:
		if(a[attribute]==current_attr):
			if(a[classification]==current_class):
				class_count+=1
	
			else:
				attr_probabilities.append(Fraction(class_count,attr_denominators[current_attr]))
				class_count=1
				current_class=a[classification]

		else:
			probabilities[current_attr]=attr_probabilities
			attr_probabilities.append(Fraction(class_count,attr_denominators[current_attr]))
			current_class=a[classification]
			current_attr=a[attribute]
			class_count=1
			attr_probabilities=[]

	attr_probabilities.append(Fraction(class_count,attr_denominators[current_attr]))
	probabilities[current_attr]=attr_probabilities
	print(probabilities)
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

	probabilities=[]

	for j in classifications:
		probabilities.append(Fraction(j,denominator))

	return probabilities

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

		
	for a_name in attr_types:
		if(a_name[1]=="categorical"):
			print("ATTRIBUTE: {0}".format(a_name[0]))
			print("===========================")
			probabilities=getClassProbabilities(data_dict,classification)
			#print(probabilities)
			attr_probabilities=getAttrProbabilities(data_dict,a_name[0],classification)
			attr_gains[a_name[0]]=calcGain(data_dict,a_name[0],attr_probabilities,probabilities)
			print("\n")
		#elif(attr_types[a_name]=="continuous")
	spiltAttr=list(attr_gains.keys())[0]
	for attr in attr_gains:
		if(attr_gains[attr]>attr_gains[spiltAttr]):
			spiltAttr=attr

	print("Splitting Attribute = {0}".format(spiltAttr))
	return spiltAttr

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
	fileHandler=open(filePath,"r")
	line=fileHandler.readline().strip()

	while(line!=""):
		attr=line.split(":")
		#print(attr)
		#attr_types[attr[0]]=attr[1]
		attr_types.append(attr)
		line=fileHandler.readline().strip()

	#print(attr_types)
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
	fileHandler=open(filePath,"r")
	line=fileHandler.readline().strip()

	while(line!=""):
		if("?" not in line):
			data_set_row=line.split(",")
			data_dict_row={}
			index=0
			for a_name in attr_types:
				data_dict_row[a_name[0]]=data_set_row[index]
				index+=1
			
			index=0
			data_dict.append(data_dict_row)
		line=fileHandler.readline().strip()

	return data_dict

	
node_count = 0
def createDecsionTree(data_dict,attr_types, parent_name, branch_name):
	global node_count
	node_count+=1
	classification=None
	new_dicts={}

	for attr_class in attr_types:
		if(attr_class[1]=="class"):
			classification=attr_class[0]

	print("remaining data", data_dict[0][classification])
	if(len(attr_types)>1):
		splitAttr=getSplittingAttribute(data_dict,attr_types)

		node_name = splitAttr+'_'+str(node_count)
		if parent_name is not None:
			print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name,branch_name), file=sys.stderr)
		
		sorted_dict=sorted(data_dict, key=lambda a: (a[splitAttr],a[classification]))
		split_attr_nodes=list(Counter(data[splitAttr] for data in data_dict).keys())
		new_dicts={node: [item for item in sorted_dict if item[splitAttr]==node] 
						for node in split_attr_nodes }
		print(new_dicts)

		for node in new_dicts:
			attr_dict=new_dicts[node]
			for d in attr_dict: 
				d.pop(splitAttr, None)

		
		attr_types=[item for item in attr_types if(item[0]!=splitAttr)]	
		print("\n=================================================================================")


		for node in new_dicts:
			attr_dict=new_dicts[node]
			createDecsionTree(attr_dict,attr_types, node_name,node)

	else:
		node_name = data_dict[0][classification]+'_'+str(node_count)
		print('"{}" -> "{}" [label="{}"];'.format(parent_name, node_name,branch_name), file=sys.stderr)
		return

"""
==============================
			MAIN 
==============================
"""
def main():
	#attr_types=getAttrTypes("./attributes.txt")
	#data_dict=getDataDict(attr_types,"./dataset.txt")
	attr_types=[["A1","categorical"],["A2","continuous"],["A3","categorical"],["Class","class"]]
	#getSplittingAttribute(data_dict,attr_types)
	print("digraph g{", file=sys.stderr)
	createDecsionTree(data_dict,attr_types,None,None)
	print("}", file=sys.stderr)




if __name__ == '__main__':
    main()