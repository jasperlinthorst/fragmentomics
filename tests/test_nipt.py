import numpy as np
import pandas as pd
import pytest
from unittest import mock

from cfstats.nipt import build_targets, calc_target_scores


def _profiles():
    labels = ["chr1_0_100", "chr1_100_200", "chr2_0_100", "chr2_100_200"]
    null = pd.Series([0.25] * 4, index=labels)
    std = pd.Series([0.01] * 4, index=labels)
    return null, std, np.array([True, True, False, False])


def test_null_profile_disfavors_both_alternatives():
    null, std, mask = _profiles()
    scores = calc_target_scores(null, null, std, mask, 0.1)
    assert scores["gain_LLR"] < 0
    assert scores["deletion_LLR"] < 0
    assert scores["Zscore"] == pytest.approx(0, abs=1e-10)


def test_gain_and_deletion_are_directional():
    null, std, mask = _profiles()
    gain = null.copy()
    gain[mask] *= 1.05
    gain /= gain.sum()
    deletion = null.copy()
    deletion[mask] *= 0.95
    deletion /= deletion.sum()
    gain_scores = calc_target_scores(gain, null, std, mask, 0.1)
    deletion_scores = calc_target_scores(deletion, null, std, mask, 0.1)
    assert gain_scores["gain_LLR"] > gain_scores["deletion_LLR"]
    assert gain_scores["Zscore"] > 0
    assert deletion_scores["deletion_LLR"] > deletion_scores["gain_LLR"]
    assert deletion_scores["Zscore"] < 0


def test_zscore_scales_with_number_of_bins():
    labels = [f"chr1_{i*100}_{(i+1)*100}" for i in range(20)]
    null = pd.Series([1.0 / 20] * 20, index=labels)
    std = pd.Series([0.01] * 20, index=labels)
    mask_2 = np.array([i < 2 for i in range(20)])
    mask_10 = np.array([i < 10 for i in range(20)])
    sample = null.copy()
    sample[mask_10] *= 1.02
    sample /= sample.sum()
    z2 = calc_target_scores(sample, null, std, mask_2, 0.1)["Zscore"]
    z10 = calc_target_scores(sample, null, std, mask_10, 0.1)["Zscore"]
    assert abs(z10) > abs(z2)


def test_ratio_scores_are_nonnegative_directional_and_unbounded():
    null, std, mask = _profiles()
    gain = null.copy()
    gain[mask] *= 1.2
    gain /= gain.sum()
    scores = calc_target_scores(gain, null, std, mask, 0.1, ratio_score=True)
    assert scores["gain_MR"] > 1
    assert scores["deletion_MR"] == 0


def test_invalid_ff_and_zero_variance_are_rejected():
    null, std, mask = _profiles()
    with pytest.raises(ValueError, match="Fetal fraction"):
        calc_target_scores(null, null, std, mask, 0, False)
    std.iloc[0] = 0
    with pytest.raises(ValueError, match="standard deviations"):
        calc_target_scores(null, null, std, mask, 0.1, False)


def test_default_targets_group_bins_by_contig():
    columns = ["chr1_0_100", "chr1_100_200", "chrX_0_100", "chrM_0_100"]
    targets = build_targets(columns)
    assert list(targets) == ["chr1", "chrX", "chrM"]
    assert targets["chr1"].tolist() == [True, True, False, False]


def test_bed_targets_use_names_and_any_overlap(tmp_path):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t90\t110\tboundary\nchr2\t0\t50\tsecond\n")
    columns = ["chr1_0_100", "chr1_100_200", "chr2_0_100"]
    targets = build_targets(columns, str(bed))
    assert list(targets) == ["boundary", "second"]
    assert targets["boundary"].tolist() == [True, True, False]


@pytest.mark.parametrize(
    "content,match",
    [
        ("chr1\t0\t100\n", "column 4"),
        ("chr1\t0\t100\tx\nchr1\t100\t200\tx\n", "unique"),
        ("chr9\t0\t100\tx\n", "does not overlap"),
    ],
)
def test_bed_target_validation(tmp_path, content, match):
    bed = tmp_path / "regions.bed"
    bed.write_text(content)
    with pytest.raises(ValueError, match=match):
        build_targets(["chr1_0_100"], str(bed))


def _workflow_reference():
    columns = ["chr1_0_100", "chr2_0_100"]
    return pd.DataFrame(
        [[0.48, 0.52], [0.50, 0.50], [0.52, 0.48]],
        index=["r1", "r2", "r3"],
        columns=columns,
    )


def test_nipt_predicts_ff_on_copy_and_writes_fixed_tsv(make_args, capsys):
    from cfstats.nipt import nipt

    args = make_args(
        samfiles=["a.bam", "b.bam"],
        referencesamples="reference.tsv",
        binsize=100,
        ff=None,
        ratio_score=True,
        regions=None,
        model="model.pickle",
        min_ff=0.01,
        max_ff=1.0,
        hf_token=None,
    )
    reference = _workflow_reference()

    def predict(copied_args, cmdline):
        assert copied_args is not args
        copied_args.binsize = 50000
        assert cmdline is False
        return np.array([0.1, 0.2])

    with mock.patch("cfstats.nipt._load_reference", return_value=reference), \
         mock.patch("cfstats.nipt.ff.ff", side_effect=predict) as predict_mock, \
         mock.patch("cfstats.nipt.bincounts.bincounts", return_value=(
             list(reference.columns), np.array([[48, 52], [52, 48]])
         )) as count_mock:
        nipt(args)

    assert args.binsize == 100
    predict_mock.assert_called_once()
    assert count_mock.call_args.args[0].bamlist is None
    lines = capsys.readouterr().out.splitlines()
    header = lines[0].split("\t")
    assert header == [
        "filename", "FF",
        "chr1_Zscore", "chr1_gain_LLR", "chr1_deletion_LLR", "chr1_gain_MR", "chr1_deletion_MR",
        "chr2_Zscore", "chr2_gain_LLR", "chr2_deletion_LLR", "chr2_gain_MR", "chr2_deletion_MR",
    ]
    assert all(len(line.split("\t")) == len(header) for line in lines[1:])
    assert [line.split("\t")[:2] for line in lines[1:]] == [["a.bam", "0.1"], ["b.bam", "0.2"]]


def test_fixed_ff_skips_prediction_and_consumes_bamlist_once(make_args, tmp_path, capsys):
    from cfstats.nipt import nipt

    bamlist = tmp_path / "bams.txt"
    bamlist.write_text("a.bam\nb.bam\n")
    args = make_args(
        samfiles=[], bamlist=str(bamlist), referencesamples="reference.tsv",
        binsize=100, ff=0.15, ratio_score=False, regions=None,
    )
    reference = _workflow_reference()
    with mock.patch("cfstats.nipt._load_reference", return_value=reference), \
         mock.patch("cfstats.nipt.ff.ff") as predict_mock, \
         mock.patch("cfstats.nipt.bincounts.bincounts", return_value=(
             list(reference.columns), np.array([[48, 52], [52, 48]])
         )) as count_mock:
        nipt(args)

    predict_mock.assert_not_called()
    assert args.samfiles == ["a.bam", "b.bam"]
    assert args.bamlist is None
    assert count_mock.call_args.args[0].nproc == args.nproc
    header = capsys.readouterr().out.splitlines()[0]
    assert "_Zscore" in header
    assert "_MR" not in header
