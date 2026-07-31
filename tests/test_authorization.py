"""Who may reach what.

Two kinds of test here, and the second matters more than the first.

The **matrix** checks the rules as they stand: an owner reaches their own
project, a stranger gets 404, a non-admin cannot wipe anything.

The **coverage** test guards the failure mode that actually bites --- not a
wrong check, but a *missing* one on a route somebody adds in six months.
Every ``/api/*`` path must be classified below. A new route with no
decision recorded fails the suite until someone makes one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autodft import accounts
from autodft.api.app import create_app
from autodft.config import Settings
from autodft.db import get_session, init_db, reset_engine
from autodft.models import Molecule


# ----------------------------------------------------------------------
# Route classification
# ----------------------------------------------------------------------

# Reachable without any credential.
PUBLIC = {"/login", "/logout"}

# Admin only. Enforced twice: at the middleware, because a missing check
# on a destructive route is unrecoverable, and again in the handler.
ADMIN_ONLY = {
    "/api/admin/circuit-breaker",
    "/api/admin/circuit-breaker/reset",
    "/api/admin/reset-preview",
    "/api/admin/reset-database",
    "/api/admin/disk-usage",
    "/api/admin/users",
    "/api/admin/users/{username}",
    "/api/admin/users/{username}/rotate-key",
    "/api/admin/projects/{name}/reassign",
}

# Filtered to the caller's own projects.
SCOPED = {
    "/api/overview",
    "/api/molecules",
    "/api/molecules/{molecule_id}",
    "/api/tasks",
    "/api/jobs",
    "/api/queue",
    "/api/projects",
    "/api/projects/{name}",
    "/api/projects/{name}/molecules-detail",
    "/api/projects/{name}/state-analysis",
    "/api/projects/{name}/state-analysis/export",
    "/api/projects/{name}/archive",
    "/api/projects/{name}/export",
    "/api/projects/{name}/jobs",
    "/api/jobs/{job_id}",
    "/api/jobs/{job_id}/download",
    # Destructive, but scoped: a user may wipe their own work, and
    # someone else's project answers 404 exactly as a read does.
    "/api/projects/{name}/wipe",
    "/api/projects/{name}/wipe-preview",
    "/api/molecules/{molecule_id}/wipe",
    "/api/molecules/{molecule_id}/wipe-preview",
    "/api/entrypoints/failed",
    "/api/submit",
    "/api/submit-batch",
}

# Deliberately shared by everyone who is logged in.
SHARED = {
    "/api/validate-smiles",   # pure SMILES parsing, touches no data
    "/api/headers",           # saved methods are shared across users
    "/api/headers/{header_id}",
    "/api/cluster",           # read-only queue depth + breaker state
    "/api/whoami",            # who am I, and what may I do
    "/api/wipe-status",       # is a deletion in flight; label admin-only
}


@pytest.fixture()
def app_env(tmp_path):
    """An app on a throwaway database, with an admin and two users."""
    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    init_db(settings)

    with get_session(settings) as session:
        admin = accounts.get_user_by_username(session, "admin")
        admin_key = accounts.rotate_api_key(session, admin)
        owner, owner_key = accounts.create_user(session, "owner")
        stranger, stranger_key = accounts.create_user(session, "stranger")
        accounts.get_or_create_project(session, owner, "screening")
        accounts.get_or_create_project(session, stranger, "screening")
        session.add(Molecule(smiles="CCO", project_name="owner/screening"))
        session.add(Molecule(smiles="CCN", project_name="stranger/screening"))
        session.commit()

    app = create_app(settings)
    with TestClient(app) as client:
        yield {
            "client": client,
            "admin": {"X-AutoDFT-API-Key": admin_key},
            "owner": {"X-AutoDFT-API-Key": owner_key},
            "stranger": {"X-AutoDFT-API-Key": stranger_key},
            # The retired shared secret. Kept in the fixture so the test
            # below can assert it is refused rather than silently dropped.
            "password": {"X-AutoDFT-Password": "password"},
        }
    reset_engine()


def _api_paths() -> set[str]:
    """Every ``/api/*`` path the application serves.

    Read off the routers rather than ``app.routes``: this FastAPI version
    wraps an included router in a lazy ``_IncludedRouter`` that does not
    expose its children, so walking the app yielded an empty set and the
    coverage assertion below passed without checking anything.
    """
    from autodft.api.routes import public_router, router

    return {
        route.path
        for source in (router, public_router)
        for route in source.routes
        if getattr(route, "path", "").startswith("/api/")
    }


class TestRouteCoverage:
    def test_the_enumeration_actually_finds_routes(self):
        """Guards the guard: an empty set would make every check below
        pass while testing nothing."""
        paths = _api_paths()
        assert len(paths) > 20, paths

    def test_every_api_route_is_classified(self):
        """A new endpoint must declare who may reach it."""
        classified = ADMIN_ONLY | SCOPED | SHARED
        unclassified = _api_paths() - classified
        assert not unclassified, (
            "These routes have no authorization decision recorded. Add each "
            "to ADMIN_ONLY, SCOPED or SHARED in this file, and enforce it in "
            f"the handler: {sorted(unclassified)}"
        )

    def test_the_classification_has_no_stale_entries(self):
        """A route that was removed should not leave a rule behind."""
        stale = (ADMIN_ONLY | SCOPED | SHARED) - _api_paths()
        assert not stale, f"Classified but no longer routed: {sorted(stale)}"

    def test_admin_routes_are_gated_at_the_middleware(self, app_env):
        """Defence in depth: the destructive routes are refused at a single
        choke point as well as in each handler, because a handler that
        forgets is not recoverable."""
        client = app_env["client"]
        for path in sorted(ADMIN_ONLY):
            if "{" in path:
                continue
            response = client.get(path, headers=app_env["owner"])
            assert response.status_code in (403, 405), f"{path} -> {response.status_code}"


class TestUnauthenticated:
    def test_the_api_refuses_without_a_credential(self, app_env):
        response = app_env["client"].get("/api/overview")
        assert response.status_code == 401

    def test_a_bad_key_is_refused(self, app_env):
        response = app_env["client"].get(
            "/api/overview", headers={"X-AutoDFT-API-Key": "adft_nope"},
        )
        assert response.status_code == 401

    def test_the_browser_is_redirected_to_login(self, app_env):
        response = app_env["client"].get("/", follow_redirects=False)
        assert response.status_code == 303
        assert "/login" in response.headers["location"]


class TestProjectVisibility:
    def test_an_owner_sees_their_own_project(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["owner"])
        assert response.status_code == 200
        assert [p["name"] for p in response.json()] == ["owner/screening"]

    def test_a_stranger_does_not_see_it(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["stranger"])
        assert [p["name"] for p in response.json()] == ["stranger/screening"]

    def test_admin_sees_everything(self, app_env):
        response = app_env["client"].get("/api/projects", headers=app_env["admin"])
        names = {p["name"] for p in response.json()}
        assert {"owner/screening", "stranger/screening"} <= names

    def test_the_shared_password_is_no_longer_a_credential(self, app_env):
        """It authenticated a crowd and resolved to admin.

        Anyone who could reach the port and knew one string held the
        destructive routes, and every submission it made was authored by
        "admin" regardless of who ran it.
        """
        response = app_env["client"].get("/api/projects", headers=app_env["password"])
        assert response.status_code == 401

    def test_no_credential_at_all_is_refused(self, app_env):
        assert app_env["client"].get("/api/projects").status_code == 401

    def test_the_password_cannot_submit(self, app_env):
        response = app_env["client"].post(
            "/api/submit", headers=app_env["password"],
            json={"smiles": "CCO", "project": "screening"},
        )
        assert response.status_code == 401

    def test_the_password_cannot_sign_in(self, app_env):
        """The login form took a blank username plus the password."""
        response = app_env["client"].post(
            "/login", data={"username": "", "password": "password"},
            follow_redirects=False,
        )
        assert response.status_code == 200
        assert "Incorrect username or key" in response.text

    def test_a_bare_name_means_my_own(self, app_env):
        """Two users have a project called 'screening'; each gets theirs."""
        client = app_env["client"]
        mine = client.get("/api/molecules?project=screening", headers=app_env["owner"])
        theirs = client.get("/api/molecules?project=screening", headers=app_env["stranger"])
        assert [m["smiles"] for m in mine.json()] == ["CCO"]
        assert [m["smiles"] for m in theirs.json()] == ["CCN"]

    def test_a_qualified_name_addresses_the_project_in_a_url(self, app_env):
        """`owner:project`, not `owner/project`.

        A percent-encoded slash is normalised back to a separator before
        routing, so `/api/projects/owner%2Fscreening` never reaches the
        handler -- and a `/api/projects/{owner}/{name}` route would collide
        with `/api/projects/{name}/export` for any project named "export".
        This test exists because the first version of it asserted a 404 and
        passed for exactly the wrong reason.
        """
        client = app_env["client"]
        assert client.get(
            "/api/projects/owner:screening", headers=app_env["owner"],
        ).status_code == 200
        # The forms that cannot work must not silently appear to.
        for unusable in ("/api/projects/owner%2Fscreening",
                         "/api/projects/owner/screening"):
            assert client.get(unusable, headers=app_env["owner"]).status_code == 404

    def test_reaching_into_another_namespace_is_a_404(self, app_env):
        """404 and not 403: a 403 would confirm the project exists."""
        response = app_env["client"].get(
            "/api/projects/stranger:screening", headers=app_env["owner"],
        )
        assert response.status_code == 404

    def test_admin_can_address_any_namespace(self, app_env):
        client = app_env["client"]
        for name in ("owner:screening", "stranger:screening"):
            assert client.get(
                "/api/projects/" + name, headers=app_env["admin"],
            ).status_code == 200

    def test_a_bare_name_admin_typed_is_refused_when_ambiguous(self, app_env):
        """Both users have 'screening'. Falling back to the admin namespace
        would silently address the wrong project."""
        response = app_env["client"].get(
            "/api/projects/screening", headers=app_env["admin"],
        )
        assert response.status_code == 409
        assert "ambiguous" in response.json()["detail"]


class TestMoleculeVisibility:
    def test_a_stranger_cannot_read_someone_elses_molecule(self, app_env):
        client = app_env["client"]
        mine = client.get("/api/molecules", headers=app_env["owner"]).json()
        molecule_id = mine[0]["id"]

        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["owner"],
        ).status_code == 200
        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["stranger"],
        ).status_code == 404
        assert client.get(
            f"/api/molecules/{molecule_id}", headers=app_env["admin"],
        ).status_code == 200

    def test_listings_are_filtered(self, app_env):
        client = app_env["client"]
        assert len(client.get("/api/molecules", headers=app_env["owner"]).json()) == 1
        assert len(client.get("/api/molecules", headers=app_env["admin"]).json()) == 2


class TestSubmission:
    def test_the_author_is_the_caller_and_not_negotiable(self, app_env):
        """A provenance label anyone may set is not provenance."""
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["owner"],
            json={"smiles": "CCO", "project": "screening", "author": "somebody-else"},
        )
        assert response.status_code == 200
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            from sqlmodel import select
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            metadata = _json.loads(entry.request_metadata)
        assert metadata["project_author"] == "owner"
        assert metadata["project_name"] == "owner/screening"
        # Echoed back, so a script can tell where its work landed without
        # a second request -- the name it sent was bare.
        assert response.json()["project"] == "owner/screening"
        assert response.json()["author"] == "owner"

    def test_admin_gets_no_exemption(self, app_env):
        """Admin is a caller like any other, not a free-text label.

        An exemption for admin is an exemption for whoever holds the
        shared dashboard password, which is the one credential that is
        not personal.
        """
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["admin"],
            json={"smiles": "CCO", "project": "onbehalf", "author": "MHT/NHO"},
        )
        assert response.status_code == 200
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            metadata = _json.loads(entry.request_metadata)
        assert metadata["project_author"] == "admin"
        assert response.json()["author"] == "admin"

    def test_submitting_to_a_name_someone_else_owns_creates_my_own(self, app_env):
        """With namespaces this is the natural reading, and it removes a
        whole class of accidental cross-writes."""
        response = app_env["client"].post(
            "/api/submit",
            headers=app_env["stranger"],
            json={"smiles": "CCO", "project": "screening"},
        )
        assert response.status_code == 200
        assert response.json()["project"] == "stranger/screening"
        with get_session() as session:
            from autodft.models import CalculationEntrypoint
            import json as _json
            entry = session.get(CalculationEntrypoint, response.json()["id"])
            assert _json.loads(entry.request_metadata)["project_name"] == "stranger/screening"


class TestDestructiveRoutes:
    @pytest.mark.parametrize("path,method", [
        ("/api/admin/reset-database", "post"),
        ("/api/admin/circuit-breaker/reset", "post"),
        ("/api/admin/users", "post"),
    ])
    def test_a_user_cannot_reach_the_admin_ones(self, app_env, path, method):
        response = getattr(app_env["client"], method)(
            path, headers=app_env["owner"], json={"confirm": "anything"},
        )
        assert response.status_code == 403

    def test_a_user_may_wipe_their_own_project(self, app_env):
        client = app_env["client"]
        preview = client.get(
            "/api/projects/owner:screening/wipe-preview", headers=app_env["owner"],
        )
        assert preview.status_code == 200
        assert preview.json()["rows"]["molecules"] == 1

        response = client.post(
            "/api/projects/owner:screening/wipe",
            headers=app_env["owner"], json={"confirm": "owner/screening"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["wiped"] is True
        assert client.get("/api/projects", headers=app_env["owner"]).json() == []

    def test_but_not_someone_elses(self, app_env):
        """404 rather than 403, for the same reason reads are: a 403 would
        confirm the project exists."""
        client = app_env["client"]
        assert client.get(
            "/api/projects/stranger:screening/wipe-preview", headers=app_env["owner"],
        ).status_code == 404
        assert client.post(
            "/api/projects/stranger:screening/wipe",
            headers=app_env["owner"], json={"confirm": "stranger/screening"},
        ).status_code == 404
        # ...and the project is still there for its owner.
        assert client.get(
            "/api/projects", headers=app_env["stranger"],
        ).json()[0]["name"] == "stranger/screening"

    def test_a_wrong_confirmation_still_refuses(self, app_env):
        response = app_env["client"].post(
            "/api/projects/owner:screening/wipe",
            headers=app_env["owner"], json={"confirm": "screening"},
        )
        assert response.status_code == 400

    def test_the_wipe_status_label_is_not_shown_to_users(self, app_env):
        """It names the project being wiped, which may be someone else's."""
        from autodft.api import admin_ops

        with admin_ops.exclusive("wipe of project 'stranger/secret'"):
            as_user = app_env["client"].get(
                "/api/wipe-status", headers=app_env["owner"],
            ).json()
            as_admin = app_env["client"].get(
                "/api/wipe-status", headers=app_env["admin"],
            ).json()

        assert as_user == {"running": True, "operation": None, "file_removal": None}
        assert "stranger/secret" in as_admin["operation"]


class TestClusterStatus:
    def test_a_user_can_see_whether_the_pipeline_is_halted(self, app_env):
        """So they can tell 'my jobs are stuck' from 'the pipeline stopped'
        without having to ask an administrator."""
        response = app_env["client"].get("/api/cluster", headers=app_env["owner"])
        assert response.status_code == 200
        assert "breaker_tripped" in response.json()

    def test_but_cannot_reset_it(self, app_env):
        response = app_env["client"].post(
            "/api/admin/circuit-breaker/reset", headers=app_env["owner"],
        )
        assert response.status_code == 403


class TestWhoami:
    def test_it_reports_the_caller(self, app_env):
        response = app_env["client"].get("/api/whoami", headers=app_env["owner"])
        assert response.json() == {
            "username": "owner", "is_admin": False, "projects": ["screening"],
        }

    def test_admin_is_flagged(self, app_env):
        response = app_env["client"].get("/api/whoami", headers=app_env["admin"])
        assert response.json()["is_admin"] is True

    def test_it_reports_what_the_caller_owns_not_what_they_can_see(self, app_env):
        """Admin sees every project; it still owns only its own. Reporting
        visibility here made admin's own projects vanish from the list."""
        client = app_env["client"]
        admin_projects = client.get("/api/whoami", headers=app_env["admin"]).json()["projects"]
        assert "screening" not in admin_projects   # that one belongs to `owner`

        client.post("/api/submit", headers=app_env["admin"],
                    json={"smiles": "CCO", "project": "mine"})
        assert client.get(
            "/api/whoami", headers=app_env["admin"],
        ).json()["projects"] == ["mine"]


class TestLogin:
    def test_username_and_key_sign_in(self, app_env, tmp_path):
        response = app_env["client"].post(
            "/login",
            data={"username": "owner", "password": app_env["owner"]["X-AutoDFT-API-Key"]},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "autodft_auth" in response.cookies or response.cookies.get("autodft_auth")

    def test_the_wrong_key_does_not(self, app_env):
        response = app_env["client"].post(
            "/login", data={"username": "owner", "password": "adft_nope"},
        )
        assert response.status_code == 200
        assert "Incorrect username or key" in response.text

    def test_a_key_belonging_to_someone_else_does_not(self, app_env):
        """Otherwise the username field would be decorative."""
        response = app_env["client"].post(
            "/login",
            data={"username": "owner",
                  "password": app_env["stranger"]["X-AutoDFT-API-Key"]},
        )
        assert response.status_code == 200
        assert "Incorrect username or key" in response.text


class TestHeaderOwnership:
    """Everyone may *use* every saved method -- a library is only useful
    shared -- but editing one silently changes what the next submission
    referencing it computes, so changes are the owner's or admin's."""

    @staticmethod
    def _create(client, headers, description):
        return client.post("/api/headers", headers=headers, json={
            "header_text": "!B3LYP def2-SVP OPT\n",
            "description": description,
            "kind": "optimization",
        })

    def test_anyone_may_create_and_it_is_theirs(self, app_env):
        response = self._create(app_env["client"], app_env["owner"], "owner's method")
        assert response.status_code == 200
        from sqlmodel import select

        from autodft.models import ComputationHeader, User
        with get_session() as session:
            header = session.get(ComputationHeader, response.json()["id"])
            user = session.exec(select(User).where(User.username == "owner")).one()
            assert header.owner_id == user.id

    def test_everyone_can_still_see_and_use_it(self, app_env):
        made = self._create(app_env["client"], app_env["owner"], "owner's method")
        listing = app_env["client"].get("/api/headers", headers=app_env["stranger"])
        assert listing.status_code == 200
        # The listing splits package defaults from stored rows.
        stored = listing.json()["custom"]
        assert made.json()["id"] in [h["id"] for h in stored]

    def test_someone_else_cannot_edit_it(self, app_env):
        made = self._create(app_env["client"], app_env["owner"], "owner's method")
        response = app_env["client"].put(
            f"/api/headers/{made.json()['id']}",
            headers=app_env["stranger"], json={"description": "hijacked"},
        )
        assert response.status_code == 403
        assert "another user" in response.json()["detail"]

    def test_someone_else_cannot_delete_it(self, app_env):
        made = self._create(app_env["client"], app_env["owner"], "owner's method")
        response = app_env["client"].delete(
            f"/api/headers/{made.json()['id']}", headers=app_env["stranger"],
        )
        assert response.status_code == 403

    def test_the_owner_can(self, app_env):
        made = self._create(app_env["client"], app_env["owner"], "owner's method")
        assert app_env["client"].put(
            f"/api/headers/{made.json()['id']}",
            headers=app_env["owner"], json={"description": "revised"},
        ).status_code == 200

    def test_admin_can_change_anyones(self, app_env):
        made = self._create(app_env["client"], app_env["owner"], "owner's method")
        assert app_env["client"].put(
            f"/api/headers/{made.json()['id']}",
            headers=app_env["admin"], json={"description": "corrected"},
        ).status_code == 200

    def test_the_seeded_defaults_belong_to_admin(self, app_env):
        """They predate accounts; ownerless would mean nobody owns them."""
        from sqlmodel import select

        from autodft.models import ComputationHeader, User
        with get_session() as session:
            admin = session.exec(select(User).where(User.username == "admin")).one()
            seeded = session.exec(select(ComputationHeader)).all()
            assert seeded, "expected the standard headers to be seeded"
            assert all(h.owner_id == admin.id for h in seeded)

    def test_a_user_cannot_edit_a_seeded_default(self, app_env):
        from sqlmodel import select

        from autodft.models import ComputationHeader
        with get_session() as session:
            first = session.exec(select(ComputationHeader)).first()
        response = app_env["client"].put(
            f"/api/headers/{first.id}",
            headers=app_env["owner"], json={"description": "changed"},
        )
        assert response.status_code == 403


def test_create_app_binds_the_database_to_its_settings(tmp_path):
    """Routes open sessions with a bare get_session(), which resolves via
    the engine singleton. If create_app does not bind it, the app reads
    the default /data/autodft instead of the configured path -- and every
    API key is rejected as unknown, which is how this was found.
    """
    from autodft.db import get_engine, reset_engine

    settings = Settings()
    settings.storage.data_path = str(tmp_path)
    reset_engine()
    create_app(settings)

    assert str(tmp_path) in str(get_engine().url)
    reset_engine()
