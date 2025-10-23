revcomptable = str.maketrans("acgtACGTRY","tgcaTGCAYR")
def revcomp(s):
    return s.translate(revcomptable)[::-1]

def allk(k,onlylexsmallest=False):
    kmers=[]
    for i in range(4**k):
        s=""
        for j in range(k):
            s+="ACGT"[int(i/(4**(k-j-1)))%4]
        if onlylexsmallest:
            if s<=revcomp(s):
                kmers.append(s)
        else:
            kmers.append(s)
    return kmers

def allkp(k,onlylexsmallest=False):
    kpmers=[]
    for i in range(2**k):
        s=""
        for j in range(k):
            s+="RY"[int(i/(2**(k-j-1)))%2]
        if onlylexsmallest:
            if s<=revcomp(s):
                kpmers.append(s)
        else:
            kpmers.append(s)
    return kpmers

#Collapse nucleotide sequence to Purine/Pyrimidine sequence
def nuc2purpyr(s):
    n2p={'A':'R','G':'R','C':'Y','T':'Y'} #R=purine / Y=Pyrimidine
    return "".join([n2p[c] for c in s])