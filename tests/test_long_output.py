"""Tests for the global --long output format (filename feature value rows)."""

import argparse
from unittest import mock

import pytest

from cfstats import fszd, csm, bincounts
from cfstats.utils import StatsWriter


class TestStatsWriter:
    def _args(self, **kw):
        return argparse.Namespace(
            name=kw.get("name", True),
            long=kw.get("long", False),
            header=kw.get("header", False),
        )

    def test_wide_default(self, capsys):
        w = StatsWriter(self._args())
        w.write("s.bam", ["A", "B"], [1, 2])
        assert capsys.readouterr().out == "s.bam\t1\t2\n"

    def test_wide_header_once(self, capsys):
        w = StatsWriter(self._args(header=True))
        w.write("s1.bam", ["A", "B"], [1, 2])
        w.write("s2.bam", ["A", "B"], [3, 4])
        assert capsys.readouterr().out == (
            "filename\tA\tB\ns1.bam\t1\t2\ns2.bam\t3\t4\n")

    def test_long(self, capsys):
        w = StatsWriter(self._args(long=True))
        w.write("s.bam", ["A", "B"], [1, 2])
        assert capsys.readouterr().out == "s.bam\tA\t1\ns.bam\tB\t2\n"

    def test_long_header_once(self, capsys):
        w = StatsWriter(self._args(long=True, header=True))
        w.write("s1.bam", ["A"], [1])
        w.write("s2.bam", ["A"], [2])
        assert capsys.readouterr().out == (
            "filename\tfeature\tvalue\ns1.bam\tA\t1\ns2.bam\tA\t2\n")

    def test_long_noname(self, capsys):
        w = StatsWriter(self._args(long=True, name=False, header=True))
        w.write("s.bam", ["A"], [1])
        assert capsys.readouterr().out == "feature\tvalue\nA\t1\n"


class TestLongFszd:
    def test_long_rows_have_filename_feature_value(self, make_args, capsys):
        args = make_args(lower=100, upper=105, long=True, header=True)
        fszd.fszd(args, cmdline=True)
        lines = capsys.readouterr().out.strip().split("\n")
        assert lines[0] == "filename\tfeature\tvalue"
        assert len(lines) == 1 + (105 - 100)
        for line in lines[1:]:
            f, feat, _ = line.split("\t")
            assert f == args.samfiles[0]
            assert feat.isdigit()

    def test_wide_unchanged(self, make_args, capsys):
        args = make_args(lower=100, upper=105, header=True)
        fszd.fszd(args, cmdline=True)
        lines = capsys.readouterr().out.strip().split("\n")
        assert len(lines) == 2
        assert lines[0].split("\t") == ["filename"] + [str(i) for i in range(100, 105)]
        assert len(lines[1].split("\t")) == 1 + (105 - 100)

    def test_long_noname(self, make_args, capsys):
        args = make_args(lower=100, upper=103, long=True, name=False)
        fszd.fszd(args, cmdline=True)
        lines = capsys.readouterr().out.strip().split("\n")
        assert all(len(l.split("\t")) == 2 for l in lines)


class TestLongCsm:
    def test_long(self, make_args, capsys, tiny_bam):
        args = make_args(k=2, long=True)
        csm.cleavesitemotifs(args, cmdline=True)
        lines = capsys.readouterr().out.strip().split("\n")
        for line in lines:
            f, feat, val = line.split("\t")
            assert f == tiny_bam
            assert len(feat) == 2


class TestLongBincounts:
    def test_long(self, make_args, capsys, tiny_bam):
        args = make_args(binsize=1000, long=True, header=True)
        bincounts.bincounts(args, cmdline=True)
        lines = capsys.readouterr().out.strip().split("\n")
        assert lines[0] == "filename\tfeature\tvalue"
        for line in lines[1:]:
            f, feat, val = line.split("\t")
            assert f == tiny_bam
            assert feat.startswith("chr1_")


class TestCliLong:
    @pytest.mark.parametrize("command", ["fszd", "bincounts", "csm", "delfi",
                                         "5pends", "siteprofile"])
    def test_long_flag_reaches_args(self, command):
        from cfstats.__main__ import lazy_cmd, main
        import sys
        argv = ["cfstats", command, "x.bam", "--long"]
        if command == "siteprofile":
            argv = ["cfstats", command, "sites.bed", "x.bam", "--long"]
        old = sys.argv
        try:
            sys.argv = argv
            with mock.patch.object(lazy_cmd, "__call__") as call:
                main()
            assert call.call_args.args[0].long is True
        finally:
            sys.argv = old

    def test_fourier_long_still_parses(self):
        """fourier's own --long resolves the conflict; parses fine."""
        from cfstats.__main__ import lazy_cmd, main
        import sys
        old = sys.argv
        try:
            sys.argv = ["cfstats", "fourier", "x.bam", "--long"]
            with mock.patch.object(lazy_cmd, "__call__") as call:
                main()
            assert call.call_args.args[0].long_format is True
        finally:
            sys.argv = old
