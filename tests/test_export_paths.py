"""Where exports land once projects are namespaced.

Nothing exercised these paths before, which is how
``export_data/admin/screening/admin/screening.csv`` -- a file whose parent
directory is never created -- survived into a passing suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import Molecule
from autodft.paths import (
    InvalidProjectName,
    project_file_stem,
    safe_subdirectory,
)


class TestFileStem:
    @pytest.mark.parametrize("name,stem", [
        ("screening", "screening"),
        ("admin/screening", "screening"),
        ("nhoelter/Project.2024", "Project.2024"),
        ("admin:screening", "screening"),
    ])
    def test_the_stem_is_the_bare_name(self, name, stem):
        """The directory already encodes the owner; repeating it in the
        filename produced a path one level deeper than the mkdir."""
        assert project_file_stem(name) == stem

    def test_a_traversal_attempt_is_still_refused(self):
        with pytest.raises(InvalidProjectName):
            project_file_stem("admin/..")


class TestExportDirectory:
    def test_a_qualified_project_nests_one_level(self, tmp_path):
        root = tmp_path / "export_data"
        root.mkdir()
        assert safe_subdirectory(root, "admin/screening") == (
            root / "admin" / "screening"
        ).resolve()

    def test_neither_half_may_escape(self, tmp_path):
        root = tmp_path / "export_data"
        root.mkdir()
        for name in ("../x/screening", "admin/..", "../../etc/passwd"):
            with pytest.raises(InvalidProjectName):
                safe_subdirectory(root, name)


@pytest.fixture()
def client(tmp_path):
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)
    with get_session(settings) as session:
        owner, key = accounts.create_user(session, "owner")
        accounts.get_or_create_project(session, owner, "screening")
        session.add(Molecule(smiles="CCO", project_name="owner/screening"))
        session.commit()
    with TestClient(create_app(settings)) as c:
        yield c, {"X-AutoDFT-API-Key": key}, tmp_path
    reset_engine()


def _run_export_job(c, headers, response):
    """Wait for the background export the POST kicked off and return its result.

    Exports now return 202 with a job; the file lands once the worker thread
    finishes.
    """
    from autodft.api import project_jobs

    assert response.status_code == 202, response.text
    project_jobs.join_all(timeout=10)
    job_id = response.json()["job"]["id"]
    detail = c.get(f"/api/jobs/{job_id}", headers=headers).json()
    assert detail["status"] == "successful", detail
    return detail["result"]


class TestExportRoute:
    @pytest.mark.parametrize("fmt,suffix", [("csv", ".csv"), ("json", ".json")])
    def test_the_file_lands_beside_the_project_not_below_it(
        self, client, fmt, suffix,
    ):
        c, headers, tmp_path = client
        response = c.post(
            f"/api/projects/owner:screening/export?format={fmt}", headers=headers,
        )
        result = _run_export_job(c, headers, response)
        expected = tmp_path / "export_data" / "owner" / "screening" / f"screening{suffix}"
        assert result["path"] == str(expected)
        # The parent must be the directory that was actually created.
        assert expected.parent.is_dir()

    def test_raw_files_go_under_the_project_directory(self, client):
        c, headers, tmp_path = client
        response = c.post(
            "/api/projects/owner:screening/export?format=files", headers=headers,
        )
        result = _run_export_job(c, headers, response)
        assert result["path"] == str(
            tmp_path / "export_data" / "owner" / "screening" / "files"
        )
