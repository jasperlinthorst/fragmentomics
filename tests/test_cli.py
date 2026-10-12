"""Tests for cfstats CLI (__main__.py): lazy_cmd, arg parsing, licence confirmation."""

import pickle
import sys
import os
import argparse
from io import StringIO
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# lazy_cmd
# ---------------------------------------------------------------------------

class TestLazyCmd:
    def test_import_and_call(self):
        """lazy_cmd should lazily import a module and call the named function."""
        from cfstats.__main__ import lazy_cmd
        # Use utils.revcomp as a simple target
        cmd = lazy_cmd("utils", "revcomp")
        assert cmd("AAAA") == "TTTT"

    def test_is_picklable(self):
        """lazy_cmd instances must survive pickle round-trips (multiprocessing)."""
        from cfstats.__main__ import lazy_cmd
        cmd = lazy_cmd("utils", "revcomp")
        restored = pickle.loads(pickle.dumps(cmd))
        assert restored("ACGT") == "ACGT"

    def test_bad_module_raises(self):
        from cfstats.__main__ import lazy_cmd
        cmd = lazy_cmd("nonexistent_module", "func")
        with pytest.raises(ModuleNotFoundError):
            cmd("arg")

    def test_bad_func_raises(self):
        from cfstats.__main__ import lazy_cmd
        cmd = lazy_cmd("utils", "nonexistent_func")
        with pytest.raises(AttributeError):
            cmd("arg")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgParsing:
    def _parse(self, argv):
        """Run main's parser on the given argv and return the namespace."""
        from cfstats.__main__ import main
        old_argv = sys.argv
        try:
            sys.argv = ["cfstats"] + argv
            # We can't call main() directly because it calls sys.exit on
            # parse errors; instead we test that known subcommands are accepted
            # by checking the parser doesn't raise.
            with pytest.raises(SystemExit) as exc_info:
                main()
            return exc_info.value.code
        finally:
            sys.argv = old_argv

    def test_no_args_prints_help(self, capsys):
        """cfstats with no args should print help (no SystemExit)."""
        from cfstats.__main__ import main
        old_argv = sys.argv
        try:
            sys.argv = ["cfstats"]
            main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert "cfstats" in captured.out

    def test_ff_missing_args_exits_2(self):
        """cfstats ff without required positional args should exit 2."""
        code = self._parse(["ff"])
        assert code == 2

    def test_ff_bounds_are_configurable(self):
        from cfstats.__main__ import lazy_cmd, main
        old_argv = sys.argv
        try:
            sys.argv = [
                "cfstats", "ff", "sample.bam", "--confirm-licence",
                "--min-ff", "0.02", "--max-ff", "0.75",
            ]
            with mock.patch.object(lazy_cmd, "__call__") as call:
                main()
            args = call.call_args.args[0]
            assert args.min_ff == pytest.approx(0.02)
            assert args.max_ff == pytest.approx(0.75)
        finally:
            sys.argv = old_argv

    def test_nipt_requires_reference_fasta(self):
        code = self._parse(["nipt", "reference.tsv", "sample.bam", "--ff", "0.1"])
        assert code == 2

    def test_nipt_parses_regions_ratio_score_and_fixed_ff(self):
        from cfstats.__main__ import lazy_cmd, main
        old_argv = sys.argv
        try:
            sys.argv = [
                "cfstats", "nipt", "reference.tsv", "a.bam", "b.bam",
                "--reference", "genome.fa", "--regions", "targets.bed",
                "--ratio-score", "--ff", "0.12", "--nproc", "2",
            ]
            with mock.patch.object(lazy_cmd, "__call__") as call, \
                 mock.patch("cfstats.__main__.confirm_model_licence") as licence:
                main()
            args = call.call_args.args[0]
            assert args.samfiles == ["a.bam", "b.bam"]
            assert args.regions == "targets.bed"
            assert args.ratio_score is True
            assert args.ff == pytest.approx(0.12)
            assert args.nproc == 2
            licence.assert_not_called()
        finally:
            sys.argv = old_argv

    def test_nipt_prediction_requires_model_licence(self):
        from cfstats.__main__ import lazy_cmd, main
        old_argv = sys.argv
        try:
            sys.argv = [
                "cfstats", "nipt", "reference.tsv", "sample.bam",
                "--reference", "genome.fa", "--confirm-licence",
            ]
            with mock.patch.object(lazy_cmd, "__call__") as call, \
                 mock.patch("cfstats.__main__.confirm_model_licence") as licence:
                main()
            assert call.call_args.args[0].ff is None
            licence.assert_called_once()
        finally:
            sys.argv = old_argv

    @pytest.mark.parametrize("value", ["0", "-0.1", "1.1"])
    def test_nipt_rejects_invalid_fixed_ff(self, value):
        code = self._parse([
            "nipt", "reference.tsv", "sample.bam", "--reference", "genome.fa",
            "--ff", value,
        ])
        assert code == 2

    def test_dnase1l3_missing_args_exits_2(self):
        code = self._parse(["dnase1l3"])
        assert code == 2

    @pytest.mark.parametrize("command", ["fourier", "deconv"])
    def test_gene_commands_accept_bamlist_without_positional_files(self, command):
        from cfstats.__main__ import lazy_cmd, main
        old_argv = sys.argv
        try:
            sys.argv = ["cfstats", command, "--bamlist", "samples.txt"]
            with mock.patch.object(lazy_cmd, "__call__") as call:
                main()
            args = call.call_args.args[0]
            assert args.samfiles == []
            assert args.bamlist == "samples.txt"
        finally:
            sys.argv = old_argv

    @pytest.mark.parametrize("command", ["fourier", "deconv"])
    def test_gene_commands_default_to_packaged_model(self, command):
        from cfstats.__main__ import DEFAULT_GENE_MODEL, lazy_cmd, main
        old_argv = sys.argv
        try:
            sys.argv = ["cfstats", command, "sample.bam"]
            with mock.patch.object(lazy_cmd, "__call__") as call:
                main()
            args = call.call_args.args[0]
            assert args.gfffile == DEFAULT_GENE_MODEL
            assert os.path.isfile(args.gfffile)
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# Licence confirmation
# ---------------------------------------------------------------------------

class TestLicenceConfirmation:
    def test_confirm_licence_flag_bypasses(self):
        """--confirm-licence should bypass the interactive prompt."""
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=True)
        # Should not raise or prompt
        confirm_model_licence(args)

    def test_accept_yes(self):
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=False)
        with mock.patch("builtins.input", return_value="yes"):
            confirm_model_licence(args)  # should not raise

    def test_accept_y(self):
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=False)
        with mock.patch("builtins.input", return_value="y"):
            confirm_model_licence(args)

    def test_reject_no(self):
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=False)
        with mock.patch("builtins.input", return_value="no"):
            with pytest.raises(SystemExit):
                confirm_model_licence(args)

    def test_reject_empty(self):
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=False)
        with mock.patch("builtins.input", return_value=""):
            with pytest.raises(SystemExit):
                confirm_model_licence(args)

    def test_reject_eof(self):
        from cfstats.__main__ import confirm_model_licence
        args = argparse.Namespace(confirm_licence=False)
        with mock.patch("builtins.input", side_effect=EOFError):
            with pytest.raises(SystemExit):
                confirm_model_licence(args)
