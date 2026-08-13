"""
Configuration: derived paths and the defaults that silently change results.

`trading_days_per_year` gets its own test because it is the sort of constant
that is invisible until two codebases disagree about it and their Sharpe ratios
stop being comparable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from evbt.config import Config, config


class TestDerivedPaths:
    def test_all_derived_paths_hang_off_the_configured_roots(self, tmp_path):
        cfg = Config(project_root=tmp_path, data_root=tmp_path / "data")

        assert cfg.raw_dir == tmp_path / "data" / "raw"
        assert cfg.processed_dir == tmp_path / "data" / "processed"
        assert cfg.results_dir == tmp_path / "results"
        assert cfg.fama_french_dir == tmp_path / "data" / "raw" / "fama_french"

    def test_data_root_can_move_independently_of_project_root(self, tmp_path):
        """Large stores live on another volume often enough to matter."""
        cfg = Config(project_root=tmp_path / "code", data_root=tmp_path / "bigdisk")

        assert cfg.raw_dir == tmp_path / "bigdisk" / "raw"
        assert cfg.results_dir == tmp_path / "code" / "results"

    def test_paths_are_path_objects_not_strings(self, tmp_path):
        cfg = Config(project_root=tmp_path, data_root=tmp_path / "data")
        for attribute in ("raw_dir", "processed_dir", "results_dir", "fama_french_dir"):
            assert isinstance(getattr(cfg, attribute), Path)

    def test_module_level_singleton_is_populated(self):
        assert config.raw_dir is not None
        assert config.fama_french_dir.name == "fama_french"


class TestEnvironmentOverrides:
    def test_data_root_reads_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_ROOT", str(tmp_path / "elsewhere"))
        assert Config().data_root == tmp_path / "elsewhere"

    def test_pit_store_root_reads_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PIT_STORE_ROOT", str(tmp_path / "project01"))
        assert Config().pit_store_root == tmp_path / "project01"

    def test_pit_store_defaults_to_the_sibling_project(self, monkeypatch):
        """
        Project 01 is a separate repo and is not imported — its output is read
        as a plain external dataset. The default points at where it lives in
        this workspace, and nothing requires it to be present.
        """
        monkeypatch.delenv("PIT_STORE_ROOT", raising=False)
        assert Config().pit_store_root.name == "data"
        assert "Project01" in str(Config().pit_store_root)


class TestSimulationDefaults:
    def test_trading_days_per_year_is_252(self):
        """
        Stated explicitly because Sharpe ratios computed on 252, 250 and 260
        are silently incomparable, and the difference is a few percent — large
        enough to matter in a comparison, small enough never to look wrong.
        """
        assert Config().trading_days_per_year == 252

    def test_french_library_url_is_the_canonical_host(self):
        url = Config().french_library_url
        assert url.startswith("https://")
        assert url.endswith("/")
        assert "ken.french" in url
