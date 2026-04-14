"""Integration tests for cfstats.delfi using a tiny synthetic BAM.

Note: delfi only supports cmdline=True (writes to stdout), so we capture output.
"""

import sys
from io import StringIO

import pytest

from cfstats import delfi


class TestDelfi:
    def test_produces_output(self, make_args):
        """delfi should write tab-separated values to stdout without crashing."""
        args = make_args(
            binsize=5000,
            shortlow=100,
            shortup=150,
            longlow=150,
            longup=200,
            insertissize=True,
            header=True,
        )
        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            delfi.delfi(args, cmdline=True)
        finally:
            sys.stdout = old_stdout
        output = buf.getvalue()
        assert len(output) > 0

    def test_header_contains_delfi(self, make_args):
        args = make_args(
            binsize=5000,
            shortlow=100,
            shortup=150,
            longlow=150,
            longup=200,
            insertissize=True,
            header=True,
        )
        old_stdout = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            delfi.delfi(args, cmdline=True)
        finally:
            sys.stdout = old_stdout
        lines = buf.getvalue().strip().split("\n")
        assert len(lines) >= 2  # header + data
        assert "delfi" in lines[0]
