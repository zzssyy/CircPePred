#!/usr/bin/env python
#_*_coding:utf-8_*_

from collections import Counter
import math, random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from HIAC import DPC, HIAC, PEDP
from Bio import Seq, SeqIO
from Maximum_Distance import run
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

def ESKmer(fasta, k=1):
    NA = 'ACGT'
    if k == 1:
        NA_mers = NA
    if k == 2:
        NA_mers = [NA1 + NA2 for NA1 in NA for NA2 in NA]
    if k == 3:
        NA_mers = [NA1 + NA2 + NA3 for NA1 in NA for NA2 in NA for NA3 in NA]
    if k == 4:
        NA_mers = [NA1 + NA2 + NA3 + NA4 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA]
    if k == 5:
        NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA]
    # if k == 6:
    #     NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA]
    # if k == 7:
    #     NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA]
    # if k == 8:
    #     NA_mers = [NA1 + NA2 + NA3 + NA4 + NA5 + NA6 + NA7 + NA8 for NA1 in NA for NA2 in NA for NA3 in NA for NA4 in NA for NA5 in NA for NA6 in NA for NA7 in NA for NA8 in NA]

    sequence = fasta
    sequence = sequence.replace('U', 'T')
    # sequence = sequence.replace('T', 'U')
    code = []
    myDict = {}
    for mer in NA_mers:
        myDict[mer] = 0
    sum = 0

    if k == 1:
        for index in range(len(sequence)):
            myDict[sequence[index]] += 1
            sum += 1

    if k != 1:
        if k % 2 == 0:
            median = int(k / 2)
        else:
            median = int((k + 1) / 2)

        for index1 in range(len(sequence) - k + 1):
            tuple1 = sequence[index1: index1 + median]
            for index2 in range(index1 + median, len(sequence) - (k - median - 1)):
                tuple2 = sequence[index2: index2 + (k - median)]
                myDict[tuple1 + tuple2] += 1
                sum += 1

    for tuple_pair in NA_mers:
        code.append(myDict[tuple_pair] / sum)
        
    return code


# SSM
def ssm_single(seq, ggaparray, g):
    # seq length is fix =23

    rst = np.zeros((16))
    for i in range(len(seq) - 1 - g):
        str1 = seq[i]
        str2 = seq[i + 1 + g]
        idx = ggaparray.index(str1 + str2)
        rst[idx] += 1

    for j in range(len(ggaparray)):
        rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

    return rst


# k-mer
def construct_sorfs_kmer():
    ntarr = ("A", "C", "G", "T")

    kmerArray = []

    for n in range(4):
        kmerArray.append(ntarr[n])

    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            kmerArray.append(str2)
    #############################################
    for n in range(4):
        str1 = ntarr[n]
        for m in range(4):
            str2 = str1 + ntarr[m]
            for x in range(4):
                str3 = str2 + ntarr[x]
                kmerArray.append(str3)
    # #############################################
    # for n in range(4):
    #     str1 = ntarr[n]
    #     for m in range(4):
    #         str2 = str1 + ntarr[m]
    #         for x in range(4):
    #             str3 = str2 + ntarr[x]
    #             for y in range(4):
    #                 str4 = str3 + ntarr[y]
    #                 kmerArray.append(str4)
    # ############################################
    # for n in range(4):
    #     str1 = ntarr[n]
    #     for m in range(4):
    #         str2 = str1 + ntarr[m]
    #         for x in range(4):
    #             str3 = str2 + ntarr[x]
    #             for y in range(4):
    #                 str4 = str3 + ntarr[y]
    #                 for z in range(4):
    #                     str5 = str4 + ntarr[z]
    #                     kmerArray.append(str5)
    return kmerArray


def SSM(fasta):
    g = [1,2,3]
    ssmarray = construct_sorfs_kmer()[4:20]
    code = []  
    
    sequence = fasta
    temp0 = ssm_single(sequence, ssmarray, g[0])
    temp1 = ssm_single(sequence, ssmarray, g[1])
    temp2 = ssm_single(sequence, ssmarray, g[2])
    code = temp0.tolist() + temp1.tolist() + temp2.tolist()
        
    return code

def AAC(sequence, **kw):
	# AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	AA = 'ACDEFGHIKLMNPQRSTVWY'

	count = Counter(sequence)
	for key in count:
		count[key] = count[key]/len(sequence)
	code = []
	for aa in AA:
		code.append(count[aa])
	return code


def construct_kmer():
	ntarr = ('D', 'E', 'K', 'R', 'A', 'N', 'C', 'Q', 'G', 'H', 'I', 'L', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

	kmerArray = []

	for n in range(20):
		str1 = ntarr[n]
		for m in range(20):
			str2 = str1 + ntarr[m]
			kmerArray.append(str2)
	return kmerArray


# DPC 400D
def DPC(fasta):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    sequence = fasta
    code = []
    tmpCode = [0] * 400
    for j in range(len(sequence) - 1):
        tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpCode = [i / sum(tmpCode) for i in tmpCode]
    code = code + tmpCode
        
    return code

def minSequenceLength(fasta):
	minLen = 10000000000
	if minLen > len(fasta):
		minLen = len(fasta)
	return minLen

# CKSAAP 400D
def CKSAAP(fasta, gap=1):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fasta) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    sequence = fasta
    code = []
    myDict = {}
    for pair in diPeptides:
        myDict[pair] = 0
    sum = 0
    for index1 in range(len(sequence)):
        index2 = index1 + gap + 1
        if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
            myDict[sequence[index1] + sequence[index2]] += 1
            sum += 1
    for pair in diPeptides:
        code.append(myDict[pair] / sum)
        
    return code

def ASDC(fasta):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    sequence = fasta
    code = []
    myDict = {}
    for pair in diPeptides:
        myDict[pair] = 0
    sum = 0
    for index1 in range(len(sequence)):
        for index2 in range(index1 + 1, len(sequence)):
            if sequence[index1] in AA and sequence[index2] in AA:
                myDict[sequence[index1] + sequence[index2]] += 1
                sum += 1
    for pair in diPeptides:
        code.append(myDict[pair] / sum)
    return code


def GTPC(sequence, **kw):
	group = {
		'alphaticr': 'GAVLMI',
		'aromatic': 'FYW',
		'postivecharger': 'KRH',
		'negativecharger': 'DE',
		'uncharger': 'STCPNQ'
	}

	groupKey = group.keys()
	baseNum = len(groupKey)
	triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]

	index = {}
	for key in groupKey:
		for aa in group[key]:
			index[aa] = key

	code = []
	myDict = {}
	for t in triple:
		myDict[t] = 0

	sum = 0
	for j in range(len(sequence) - 3 + 1):
		myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] = myDict[index[sequence[j]]+'.'+index[sequence[j+1]]+'.'+index[sequence[j+2]]] + 1
		sum = sum +1

	if sum == 0:
		for t in triple:
			code.append(0)
	else:
		for t in triple:
			code.append(myDict[t]/sum)

	return code


# single nucleic ggap
def g_gap_single(seq, ggaparray, g):
	# seq length is fix =23

	rst = np.zeros((400))
	for i in range(len(seq) - 1 - g):
		str1 = seq[i]
		str2 = seq[i + 1 + g]
		idx = ggaparray.index(str1 + str2)
		rst[idx] += 1

	for j in range(len(ggaparray)):
		rst[j] = rst[j] / (len(seq) - 1 - g)  # l-1-g

	return rst

def GGAP(sequence, **kw):
	kmerArray = construct_kmer()
	ggap = g_gap_single(sequence, kmerArray, 1)
	return ggap.tolist()


def QSOrder(sequence, nlag=5, w=0.1, **kw):
	dataFile = './dataset/Schneider-Wrede.txt'
	dataFile1 = './dataset/Grantham.txt'

	AA = 'ACDEFGHIKLMNPQRSTVWY'
	AA1 = 'ARNDCQEGHILKMFPSTWYV'

	DictAA = {}
	for i in range(len(AA)):
		DictAA[AA[i]] = i

	DictAA1 = {}
	for i in range(len(AA1)):
		DictAA1[AA1[i]] = i

	with open(dataFile) as f:
		records = f.readlines()[1:]
	AADistance = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance.append(array)
	AADistance = np.array(
		[float(AADistance[i][j]) for i in range(len(AADistance)) for j in range(len(AADistance[i]))]).reshape((20, 20))

	with open(dataFile1) as f:
		records = f.readlines()[1:]
	AADistance1 = []
	for i in records:
		array = i.rstrip().split()[1:] if i.rstrip() != '' else None
		AADistance1.append(array)
	AADistance1 = np.array(
		[float(AADistance1[i][j]) for i in range(len(AADistance1)) for j in range(len(AADistance1[i]))]).reshape(
		(20, 20))

	code = []
	arraySW = []
	arrayGM = []
	for n in range(1, nlag + 1):
		arraySW.append(
			sum([AADistance[DictAA[sequence[j]]][DictAA[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
		arrayGM.append(sum(
			[AADistance1[DictAA1[sequence[j]]][DictAA1[sequence[j + n]]] ** 2 for j in range(len(sequence) - n)]))
	myDict = {}
	for aa in AA1:
		myDict[aa] = sequence.count(aa)
	for aa in AA1:
		code.append(myDict[aa] / (1 + w * sum(arraySW)))
	for aa in AA1:
		code.append(myDict[aa] / (1 + w * sum(arrayGM)))
	for num in arraySW:
		code.append((w * num) / (1 + w * sum(arraySW)))
	for num in arrayGM:
		code.append((w * num) / (1 + w * sum(arrayGM)))
	return code


def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
	

def PAAC(sequence, lambdaValue=4, w=0.05, **kw):
	dataFile = './dataset/PAAC.txt'
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])

	encodings = []
	header = ['#']
	for aa in AA:
		header.append('Xc1.' + aa)
	for n in range(1, lambdaValue + 1):
		header.append('Xc2.lambda' + str(n))
	encodings.append(header)

	code = []
	theta = []
	for n in range(1, lambdaValue + 1):
		theta.append(
			sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
			len(sequence) - n))
	myDict = {}
	for aa in AA:
		myDict[aa] = sequence.count(aa)
	code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
	code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
	return code


def CTDC_Count(seq1, seq2):
	sum = 0
	for aa in seq1:
		sum = sum + seq2.count(aa)
	return sum


def CTDC(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	for p in property:
		c1 = CTDC_Count(group1[p], sequence) / len(sequence)
		c2 = CTDC_Count(group2[p], sequence) / len(sequence)
		c3 = 1 - c1 - c2
		code = code + [c1, c2, c3]
	return code


def CTDT(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
	for p in property:
		c1221, c1331, c2332 = 0, 0, 0
		for pair in aaPair:
			if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
				c1221 = c1221 + 1
				continue
			if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
				c1331 = c1331 + 1
				continue
			if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
				c2332 = c2332 + 1
		code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]

	return code


def CTDD_Count(aaSet, sequence):
	number = 0
	for aa in sequence:
		if aa in aaSet:
			number = number + 1
	cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
	cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

	code = []
	for cutoff in cutoffNums:
		myCount = 0
		for i in range(len(sequence)):
			if sequence[i] in aaSet:
				myCount += 1
				if myCount == cutoff:
					code.append((i + 1) / len(sequence) * 100)
					break
		if myCount == 0:
			code.append(0)
	return code


def CTDD(sequence, **kw):
	group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
	group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
	group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}

	groups = [group1, group2, group3]
	property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

	code = []
	for p in property:
		code = code + CTDD_Count(group1[p], sequence) + CTDD_Count(group2[p], sequence) + CTDD_Count(group3[p], sequence)

	return code

def get_aas(posi_file, nega_file, posi_file_name, nega_file_name):
    aas_posi_samples = []
    with open(posi_file, "r") as lines:
        for data in lines:
            line = data.strip()
            result = str()
            if line[0] == '>':
                name = line
            else:
                fasta = Seq.Seq(line)
                result = name + '\n' + str(fasta.translate(to_stop=True)) + '\n'
                aas_posi_samples.append(result)
                    
    aas_nega_samples = []
    with open(nega_file, "r") as lines:
        for data in lines:
            line = data.strip()
            fasta = str()
            if line[0] == '>':
                name = line
            else:
                fasta = Seq.Seq(line)
                result = name + '\n' + str(fasta.translate(to_stop=True)) + '\n'
                aas_nega_samples.append(result)
   
    posi_file_name = posi_file_name
    with open(posi_file_name, 'w') as posi_file:   
        posi_file.writelines(aas_posi_samples)
        posi_file.close()
    
    nega_file_name = nega_file_name
    with open(nega_file_name, 'w') as nega_file:
        nega_file.writelines(aas_nega_samples)
        nega_file.close()
    
    return posi_file_name, nega_file_name


def get_sorfs_dataset(posi_file, nega_file):
    posi_samples = []
    indexs = []
    fea_name = []
    flag= True
    with open(posi_file, "r") as lines:
        for data in lines:
            line = data.strip()
            if line[0] == '>':
                name = line[1:]
            else:
                if flag:
                    num = 0
                    sequence = line
                    es3mer_fea = ESKmer(sequence,k=3)
                    num += len(es3mer_fea)
                    indexs.append(num)
                    fea_name.append('es3mer')
                    es4mer_fea = ESKmer(sequence,k=4)
                    num += len(es4mer_fea)
                    indexs.append(num)
                    fea_name.append('es4mer')
                    es5mer_fea = ESKmer(sequence,k=5)
                    num += len(es5mer_fea)
                    indexs.append(num)
                    fea_name.append('es5mer')
                    ssm_fea = SSM(sequence)
                    num += len(ssm_fea)
                    indexs.append(num)
                    fea_name.append('ssm')
                    posi_sample = es3mer_fea + es4mer_fea + es5mer_fea + ssm_fea + [1]
                    posi_samples.append(posi_sample)
                    flag = False
                else:
                    sequence = line
                    es3mer_fea = ESKmer(sequence,k=3)
                    es4mer_fea = ESKmer(sequence,k=4)
                    es5mer_fea = ESKmer(sequence,k=5)
                    ssm_fea = SSM(sequence)
                    posi_sample = es3mer_fea + es4mer_fea + es5mer_fea + ssm_fea + [1]
                    posi_samples.append(posi_sample)

    nega_samples = []
    with open(nega_file, "r") as lines:
        for data in lines:
            line = data.strip()
            if line[0] == '>':
                name = line[1:]
            else:
                sequence = line
                es3mer_fea = ESKmer(sequence,k=3)
                es4mer_fea = ESKmer(sequence,k=4)
                es5mer_fea = ESKmer(sequence,k=5)
                ssm_fea = SSM(sequence)
                nega_sample = es3mer_fea + es4mer_fea + es5mer_fea + ssm_fea + [0]
                nega_samples.append(nega_sample)

    # random.shuffle(posi_samples)
    # random.shuffle(nega_samples)
    print(indexs)
    print(fea_name)
    posi_samples = np.array(posi_samples)
    nega_samples = np.array(nega_samples)
    mn = preprocessing.MinMaxScaler()
    posi_samples, posi_y = mn.fit_transform(posi_samples[:,:-1]), posi_samples[:,-1]
    posi_samples = np.concatenate((posi_samples, posi_y[:,np.newaxis]), axis=1)
    nega_samples, nega_y = mn.fit_transform(nega_samples[:,:-1]), nega_samples[:,-1]
    nega_samples = np.concatenate((nega_samples, nega_y[:,np.newaxis]), axis=1)
    return posi_samples, nega_samples, indexs, fea_name


def get_aas_dataset(posi_file, nega_file):
    posi_samples = []
    indexs = []
    fea_name = []
    flag = True
    with open(posi_file, "r") as lines:
        for data in lines:
            line = data.strip()
            if line[0] == '>':
                name = line[1:]
            else:
                if flag:
                    num = 0
                    sequence = line
                    aac_fea = AAC(sequence)
                    num += len(aac_fea)
                    indexs.append(num)
                    fea_name.append('aac')
                    dpc_fea = DPC(sequence)
                    num += len(dpc_fea)
                    indexs.append(num)
                    fea_name.append('dpc')
                    cks_fea = CKSAAP(sequence)
                    num += len(cks_fea)
                    indexs.append(num)
                    fea_name.append('cks')
                    asdc_fea = ASDC(sequence)
                    num += len(asdc_fea)
                    indexs.append(num)
                    fea_name.append('asdc')
                    gga_fea = GGAP(sequence)
                    num += len(gga_fea)
                    indexs.append(num)
                    fea_name.append('gga')
                    qso_fea = QSOrder(sequence)
                    num += len(qso_fea)
                    indexs.append(num)
                    fea_name.append('qso')
                    gtp_fea = GTPC(sequence)
                    num += len(gtp_fea)
                    indexs.append(num)
                    fea_name.append('gtp')
                    paac_fea = PAAC(sequence)
                    num += len(paac_fea)
                    indexs.append(num)
                    fea_name.append('paac')
                    c_fea = CTDC(sequence)
                    t_fea = CTDT(sequence)
                    d_fea = CTDD(sequence)
                    ctd_fea = c_fea + t_fea + d_fea
                    num += len(ctd_fea)
                    indexs.append(num)
                    fea_name.append('ctd')
                    posi_sample = aac_fea + dpc_fea + cks_fea + asdc_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [1]
                    # posi_sample = aac_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [1]
                    posi_samples.append(posi_sample)
                    flag = False
                else:
                    sequence = line
                    aac_fea = AAC(sequence)
                    dpc_fea = DPC(sequence)
                    cks_fea = CKSAAP(sequence)
                    asdc_fea = ASDC(sequence)
                    gga_fea = GGAP(sequence)
                    qso_fea = QSOrder(sequence)
                    gtp_fea = GTPC(sequence)
                    paac_fea = PAAC(sequence)
                    c_fea = CTDC(sequence)
                    t_fea = CTDT(sequence)
                    d_fea = CTDD(sequence)
                    ctd_fea = c_fea + t_fea + d_fea
                    posi_sample = aac_fea + dpc_fea + cks_fea + asdc_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [1]
                    # posi_sample = aac_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [1]
                    posi_samples.append(posi_sample)

    nega_samples = []
    with open(nega_file, "r") as lines:
        for data in lines:
            line = data.strip()
            if line[0] == '>':
                name = line[1:]
            else:
                sequence = line
                aac_fea = AAC(sequence)
                dpc_fea = DPC(sequence)
                cks_fea = CKSAAP(sequence)
                asdc_fea = ASDC(sequence)
                gga_fea = GGAP(sequence)
                qso_fea = QSOrder(sequence)
                gtp_fea = GTPC(sequence)
                paac_fea = PAAC(sequence)
                c_fea = CTDC(sequence)
                t_fea = CTDT(sequence)
                d_fea = CTDD(sequence)
                ctd_fea = c_fea + t_fea + d_fea
                nega_sample = aac_fea + dpc_fea + cks_fea + asdc_fea + gga_fea + qso_fea + gtp_fea + paac_fea + ctd_fea + [0]
                nega_samples.append(nega_sample)

# 	random.shuffle(posi_samples)
# 	random.shuffle(nega_samples)
    print(indexs)
    print(fea_name)
    posi_samples = np.array(posi_samples)
    nega_samples = np.array(nega_samples)
    mn = preprocessing.MinMaxScaler()
    posi_samples, posi_y = mn.fit_transform(posi_samples[:,:-1]), posi_samples[:,-1]
    posi_samples = np.concatenate((posi_samples, posi_y[:,np.newaxis]), axis=1)
    nega_samples, nega_y = mn.fit_transform(nega_samples[:,:-1]), nega_samples[:,-1]
    nega_samples = np.concatenate((nega_samples, nega_y[:,np.newaxis]), axis=1)
    return posi_samples, nega_samples, indexs, fea_name

def conn_shuf_split_dataset(posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples):
    sorfs_f_len = posi_sorfs_samples.shape[1]
    posi_samples = np.hstack((posi_sorfs_samples, posi_aas_samples))
    nega_samples = np.hstack((nega_sorfs_samples, nega_aas_samples))
    random.shuffle(posi_samples)
    random.shuffle(nega_samples)
    posi_sorfs_samples, posi_aas_samples = posi_samples[:, :sorfs_f_len], posi_samples[:, sorfs_f_len:]
    print(posi_sorfs_samples[:5])
    nega_sorfs_samples, nega_aas_samples = nega_samples[:, :sorfs_f_len], nega_samples[:, sorfs_f_len:]
    print(nega_sorfs_samples[:5])
    return posi_sorfs_samples, nega_sorfs_samples, posi_aas_samples, nega_aas_samples

def resampling(posi_samples, nega_samples):
    X = np.vstack((posi_samples[:,:-1],nega_samples[:,:-1]))
    y = np.array(posi_samples[:,-1].tolist() + nega_samples[:,-1].tolist())
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    X_new_y = np.concatenate((X,y[:,np.newaxis]), axis=1)
    posi_samples, nega_samples = X_new_y[np.where(y==1)[0]], X_new_y[np.where(y==0)[0]]
    return posi_samples, nega_samples


def feature_selection(posi_samples, nega_samples, feature_cate):
    m = len(posi_samples)
    posi_samples, nega_samples = resampling(posi_samples, nega_samples)
    X = np.vstack((posi_samples[:,:-1],nega_samples[:,:-1]))
    y = np.array(posi_samples[:,-1].tolist() + nega_samples[:,-1].tolist())
    X_new, Eachfeature = run(X, y, feature_cate)
    print(X_new[:3])
    print(X_new.shape)
    X_new_y = np.concatenate((X_new, y[:,np.newaxis]), axis=1)
    posi_samples, nega_samples = X_new_y[np.where(y==1)[0]], X_new_y[np.where(y==0)[0]]
    return posi_samples[:m], nega_samples, Eachfeature
	
#split by different clustering methods

def selective_addition_mechanism(nega_train_data):
    with_label = True
    k = 20# the number of nearest neighbors, parameter k in HIAC
    T = 0.5# parameter T in HIAC
    d = 4# the d in paper HIAC
    threshold = 2550# the weight threshold to clip invalid-neighbors
    ########################normalization###################################
    data_without_nml = nega_train_data.copy()
    if with_label:
        labels = nega_train_data[:, -1]
        labels = np.array(labels, dtype=np.int32)
        nega_train_data = nega_train_data[:, :-1]
    for j in range(nega_train_data.shape[1]):
        max_ = max(nega_train_data[:, j])
        min_ = min(nega_train_data[:, j])
        if max_ == min_:
            continue
        for i in range(nega_train_data.shape[0]):
            nega_train_data[i][j] = (nega_train_data[i][j] - min_) / (max_ - min_)
    distanceTGP, threshold = HIAC.TGP(nega_train_data, k, threshold)  # we can determine the threshold，and return the weight matrix
    neighbor_index = HIAC.prune(nega_train_data, k, threshold, distanceTGP) # clip invalid-neighbors based on the weight threshold and the decision-graph,
                                                            # and then return the index matrix which records the valid-neighbors index of object i
                                                            # for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1,
                                                            # else neighbor_index[i][j] is the index of object j
                                                            # its necessary for you to know that we only need K-nearest-neighbor of each object,
                                                            # so,

    for i in range(d): # ameliorated the dataset by d time-segments
        bata = HIAC.shrink(nega_train_data, k, T, neighbor_index)
        nega_train_data = bata
    
    return np.concatenate((nega_train_data, labels[:,np.newaxis]), axis=1)

def spliting_by_clustering(nega_train_data, group_num, method="kmeans"):
    nega_train_data = np.array(nega_train_data)
    if method == 'kmeans':
        cluster = KMeans(n_clusters=group_num)
        cluster.fit(nega_train_data)
        labels = cluster.labels_
    if method == 'AgglomerativeClustering':
        cluster = AgglomerativeClustering(n_clusters=group_num)
        cluster.fit(nega_train_data)
        labels = cluster.labels_
    if method == "PEDP":
        labels = PEDP.PEDP(nega_train_data, group_num)
        
    nega_new_train_data = {}
    for i in range(group_num):
        group_list = []
        nega_new_train_data[i] = group_list
		
    count = 0
    for label in labels:
        nega_new_train_data[label].append(nega_train_data[count])
        count = count + 1
    return nega_new_train_data

def sampling_from_clusters(nega_new_train_data, posi_num, nega_num):
	X_nega_train = []
	nega_train_data ={}
	nega_train_num = 0
	rest_nega = []
	for key in nega_new_train_data:
		sample_num = round(len(nega_new_train_data[key])/nega_num * posi_num)
		group_samples = nega_new_train_data[key][:sample_num]
		nega_train_data[key] = nega_new_train_data[key][sample_num:]
		nega_train_num += len(nega_train_data[key])
		rest_nega += nega_train_data[key]
		X_nega_train = X_nega_train + group_samples
	rest_nega = np.array(rest_nega)
	return X_nega_train,rest_nega,nega_train_num

def get_fea_name_dict(f_name, Eachfeature):
    value = 0
    key = ''
    f_nef = {}
    for i,j in zip(f_name, Eachfeature):
        h = value
        key = i
        value += len(j)
        values = (h, value)
        f_nef[key] = values
    return f_nef