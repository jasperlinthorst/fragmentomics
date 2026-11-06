"""Regression tests for indel read counting in cfstats impute."""

import pysam
import pytest

from cfstats.impute import core as impute_core


def _build_bam(path, reads):
    header = {"HD": {"VN": "1.0", "SO": "coordinate"},
              "SQ": [{"LN": 1000, "SN": "chr1"}]}
    with pysam.AlignmentFile(path, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.index(path)


def _make_read(name, start, seq, quals, cigar, mapq=60, tlen=100):
    r = pysam.AlignedSegment()
    r.query_name = name
    r.reference_id = 0
    r.reference_start = start
    r.query_sequence = seq
    r.query_qualities = quals
    r.cigartuples = cigar
    r.mapping_quality = mapq
    r.template_length = tlen
    r.is_unmapped = False
    r.is_secondary = False
    r.is_duplicate = False
    return r


def test_deletion_read_counted_as_alt_at_anchor_base(tmp_path):
    """A deletion read must be counted as alt at the upstream anchor base.

    VCF represents a 3bp deletion as e.g. REF=ATCT, ALT=A at the anchor
    position.  pysam reports the deletion length as a negative indel at the
    anchor base (the last matching base before the D cigar op), so reads
    covering that base with pcr.indel == -del_len are the alt allele.
    """
    bam = str(tmp_path / "test.bam")
    reads = [
        # Reference read: covers anchor base, no indel
        _make_read("ref", 5, "AAAAATCTGG", [30] * 10, [(0, 10)]),
        # Deletion read: 5M, 3D, 5M -> deletes TCT at ref 10-12
        _make_read("del", 5, "AAAAAGGGGG", [30] * 10, [(0, 5), (2, 3), (0, 5)]),
    ]
    _build_bam(bam, reads)

    # POS=10 is the anchor base (the A before the deleted TCT)
    variants = [("F508del", 10, "ATCT", "A")]
    R = impute_core.getR(bam, "chr1", variants)

    observations = [(i, a, q)
                    for (_, _), (obs, _, _, _, _) in R.items()
                    for i, a, q in obs]
    assert len(observations) == 2
    alt_count = sum(1 for _, a, _ in observations if a == 1)
    ref_count = sum(1 for _, a, _ in observations if a == 0)
    assert ref_count == 1
    assert alt_count == 1
