#!/usr/bin/env python 
""" 
	file.py

	Some functions for file handling
		
	Matthew Hayes
	May 20 2008, Geneva
"""

import os 

def getlines(fn): 
	"""
		ubersimple function. 
		gets all the lines 
	"""
	fh = open(fn, "r")
	d  = fh.readlines()
	fh.close()
	return d




def zap(fn): 
	"""
		ubersimple function. 
		if fn exists, get rid of it.
	"""
	if os.path.exists(fn): os.remove(fn)	




def ReadStripTok(fn, commchar='#'):
	"""
		Function that reads an ascii file, strips out comments and whitespace,
		converts to lowercase, tokenises, and returns the list.
	"""
	cont=[]
	h=open(fn, 'r')
	for line in h.readlines():
		if commchar != None:
			icom=line.find(commchar)   #find comments and strip
			if icom != -1:
				line=line[:icom]
		line=line.strip()      #dump leading and trailing whitespace
		if len(line)>0:
			#cont.append(line.lower().split())
			cont.append(line.split())
	h.close()
	return cont



def uncomment_reader(fn):
	fh=open(fn, "r")
	cont=fh.readlines()
	fh.close()
	dat= [ l.replace("\n", "").split("#")[0] for l in cont if l[0]!="#" ]
	return dat
	

