from Bio import SeqIO
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def get_gc_content_from_fasta(fasta_path, chrom, start, end):
    # Load the chromosome sequence from the fasta file
    # Assumes chromosome names in fasta are like 'chr1', 'chr2', etc.
    if not hasattr(get_gc_content_from_fasta, "seqs"):
        # Cache loaded sequences for efficiency
        get_gc_content_from_fasta.seqs = {rec.id: rec.seq for rec in SeqIO.parse(fasta_path, "fasta")}
    seqs = get_gc_content_from_fasta.seqs
    seq = seqs.get(chrom)
    if seq is None:
        return np.nan
    subseq = seq[int(start):int(end)]
    gc = (subseq.count("G") + subseq.count("C")) / len(subseq) if len(subseq) > 0 else np.nan
    return gc

# Parse columns and compute GC content for each bin
def get_gc_content(dfcnt):
    gc_content = []
    for col in dfcnt.columns:
        parts = col.split('_')
        chrom = parts[0]
        if len(parts) == 4:  # e.g., chr1_0_1000000
            start, end = parts[1], parts[2]
        elif len(parts) == 3:  # e.g., chr1_0_1000000
            start, end = parts[1], parts[2]
        else:
            gc_content.append(np.nan)
            continue
        gc = get_gc_content_from_fasta("hg38flat.fa", chrom, start, end)
        gc_content.append(gc)

    return pd.Series(gc_content, index=dfcnt.columns, name="gc_content")

def gc_correct_counts(counts_df, gc_content):
    # Fit a loess/lowess or polynomial regression for each sample to correct for GC bias

    corrected = pd.DataFrame(index=counts_df.index, columns=counts_df.columns)
    i=0
    for idx, row in counts_df.iterrows():
        mask = (~row.isna()) & (~gc_content.isna())
        if mask.sum() < 10:
            corrected.loc[idx] = row
            continue
        fitted = lowess(row[mask], gc_content[mask], frac=0.1, return_sorted=False)
        corrected.loc[idx, mask] = row[mask] / fitted * np.nanmedian(row[mask])
        corrected.loc[idx, ~mask] = np.nan
        i+=1
        if i%1000==0:
            print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),"At %d ..."%i)
    return corrected

def get_N_content_from_fasta(fasta_path, chrom, start, end):
    # Load the chromosome sequence from the fasta file
    # Assumes chromosome names in fasta are like 'chr1', 'chr2', etc.
    if not hasattr(get_gc_content_from_fasta, "seqs"):
        # Cache loaded sequences for efficiency
        get_gc_content_from_fasta.seqs = {rec.id: rec.seq for rec in SeqIO.parse(fasta_path, "fasta")}
    seqs = get_gc_content_from_fasta.seqs
    seq = seqs.get(chrom)
    if seq is None:
        return np.nan
    subseq = seq[int(start):int(end)]
    Nc = subseq.count("N") / len(subseq) if len(subseq) > 0 else np.nan
    return Nc