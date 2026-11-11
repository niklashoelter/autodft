# AutoDFT

Automated DFT calculation pipeline with a SQLite backend, SLURM
integration, and a FastAPI dashboard. Submit molecules by SMILES; the
controller drives them through conformer search → optimisation →
single-point (+ optional vertical excitations / ox / red) using ORCA;
results are extractable as CSV / JSON or as raw ORCA files.

## At a glance

```
   submit (CLI / REST / python)
        │
        ▼
 [CalculationEntrypoint] ──► entrypoint_processor ──► Molecule / State / Task
                                                          │
                                                          ▼
                                                   state_machine
                                                   ├─ build inputs
                                                   ├─ submit to SLURM
                                                   ├─ poll status
                                                   └─ parse outputs
                                                          │
                                                          ▼
                                                   PipelineExtractor
                                                   ├─ export CSV / JSON
                                                   └─ export raw files
```

Parsing a successful job's output (`parse outputs`, above) also stores
the result once, in `job_results`; readers — analyses, exports, the
Photophysics page, ESD rate-job inputs — use the stored record and fall
back to the job's files only when there is none. A newly judged
optimisation's record also says whether its geometry kept the SMILES's
bonds; a broken one is badged in the dashboard and never used for ESD
(see [docs/PHOTOPHYSICS.md](docs/PHOTOPHYSICS.md#broken-bonds)).

All persistent state lives in **one directory** — the `data_path`. Per
the production config (`config/reaction.toml`):

```
/mnt/share/dft_calculations/autodft_data/
├── autodft.db        # SQLModel / SQLite database (WAL)
├── comp_data/        # per-molecule SLURM working directories
└── export_data/      # CSV / JSON / raw-file exports, as <owner>/<project>/
```

The same path **must be reachable by both the controller and every SLURM
compute node** (NFS / Lustre / BeeGFS).

> **API reference.** Every REST endpoint with field-level docs and
> example bodies lives in [`docs/API.md`](docs/API.md).

---

## Install

Dependencies are declared as ranges in `pyproject.toml`; there is no
lockfile, so an install resolves to the newest compatible versions.
Fastest path is `uv`:

```bash
# from the project root
uv venv .venv
uv pip install -e ".[dev]"   # dev extras bring pytest + httpx
source .venv/bin/activate
```

On the controller, `.venv` is a symlink to `/tmp/.venv_autodft` (no venvs on
NFS). `/tmp` is wiped on reboot; rebuild it with `uv venv /tmp/.venv_autodft`
and `uv pip install --python /tmp/.venv_autodft/bin/python -r .git/venv_pins.txt`
(the frozen package list) before restarting the controller.

Without `uv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .             # uses pyproject.toml
# optional: pip install -e ".[dev]" for pytest etc.
```

**Required on the controller** (declared in `pyproject.toml`):

* RDKit — converts SMILES → initial 3-D geometry. Hard requirement; the
  pipeline **fails loud** rather than submitting a placeholder structure
  if RDKit (and OpenBabel as fallback) are unavailable. Bad SMILES land
  on `/api/entrypoints/failed` with the error visible in the dashboard.

**Required on the compute nodes:**

* **ORCA** — quantum-chemistry engine. The cluster's binary lives at
  `/mnt/share/public/software/orca-6.1.1-gxtb/orca` (this is the build
  that includes g-xTB; the seeded GOAT g-xTB header relies on it).
  Point `[orca].path` at the absolute path — no `module load` happens.
* **SLURM** — `sbatch`, `squeue`, `sacct`.
* **NBO 7** (optional) — set `[orca].nbo_exe` and the per-job submit
  script will export `NBOEXE` so `%nbo` blocks work.

---

## Configure

The loader merges three layers, highest-priority last:

1. `config/default.toml` — framework defaults (shipped, don't edit)
2. the file you pass via `--config`. Every command that writes or serves the
   database requires it: `run`, `submit submit`/`submit-batch` and every
   `admin` command but `progress`. Without it they exit before loading
   anything. The read-only `status *` and `admin progress` default to
   `config/reaction.toml` of the checkout they run from (beside the `autodft`
   package), whatever the current directory. Every command but `run` exits 1,
   naming the host, before opening the database when
   `<data_path>/controller.lock` names another host (SQLite on NFS: one host
   only); without a lock file it proceeds. `run` refuses through the lock itself
3. environment variables (`AUTODFT_*`)

Loading is strict: a `--config` path that does not exist, a key or section
the loader does not know (e.g. `stall_minute`, `[pipeline.optimisation]`) and a
value of the wrong type (e.g. `port = "8085"`) each stop the command with an
error naming the path or every offending key, instead of falling back to
defaults. An integer is accepted where a float is expected.

For production, the entire config is anchored on `data_path` and the
ORCA section:

```toml
# config/reaction.toml
[storage]
data_path = "/mnt/share/dft_calculations/autodft_data"
# database, comp_data/, export_data/ are derived from this

[pipeline]
max_simultaneous_entrypoints = 50
queue_slots_per_priority     = 10   # priority p -> p*10 waiting SLURM jobs
max_unsubmitted_jobs         = 500  # DB backlog ceiling before expansion pauses
loop_interval_seconds        = 60
max_attempts                 = 3
# The two submission-throttle keys are not in the shipped file; the values
# below are the defaults that apply when they are absent.
max_submission_seconds_per_tick = 30  # yield after this much time in sbatch
max_submissions_per_tick     = 0   # 0 = no count limit; the queue cap throttles

[slurm]
partition = "CPU"
nice      = 1000

[orca]
# Absolute path to the ORCA binary on compute nodes. Use "orca" only
# if a module system has already placed it on PATH (no module system
# is used in this deployment).
path       = "/mnt/share/public/software/orca-6.1.1-gxtb/orca"
# Second argument to ORCA — controls MPI binding. Empty string disables.
extra_args = "--bind-to none"
# Optional script sourced before ORCA runs (e.g. to select an Open MPI install).
# mpi_env  = "/path/to/openmpi/env.sh"
# Optional NBO 7 executable (exported as NBOEXE when set).
nbo_exe    = "/mnt/share/public/software/nbo7/bin/nbo7.i8.exe"
# Parent of the per-job scratch dir. Empty string runs ORCA inside
# the shared job dir without staging through /tmp.
tmp_dir    = "/tmp"
# Stall watchdog: kill ORCA once nothing in its scratch dir changed for
# this many minutes; the job fails as "Stalled" and is resubmitted with
# the same resources. 0 (the default) runs ORCA without a watchdog.
# (A job that segfaulted without ORCA terminating normally fails as
# "Crashed" and is resubmitted unchanged the same way, watchdog or not.)
stall_minutes       = 240
# Copy output.out back from scratch this often while ORCA runs (needs
# the watchdog), so running jobs show progress. 0 disables.
sync_output_minutes = 10

[api]
enabled = true
host    = "0.0.0.0"
port    = 8085

[security]
# There is no shared password: access is a username plus that account's
# API key. `session_secret` only signs the session cookie and is
# generated into <data_path>/.session_secret when left unset.
# session_secret         = ""
session_lifetime_seconds = 604800   # 7 days
```

Override individual values without editing the file:

| Variable                | What it sets                              |
| ----------------------- | ----------------------------------------- |
| `AUTODFT_DATA_PATH`     | `storage.data_path`                       |
| `AUTODFT_COMP_DATA`     | `storage.comp_data_path`                  |
| `AUTODFT_EXPORT_DATA`   | `storage.export_data_path`                |
| `AUTODFT_DB_URL`        | `database.url` (skip data_path derivation) |
| `AUTODFT_PARTITION`     | `slurm.partition`                         |
| `AUTODFT_API_PORT`      | `api.port`                                |
| `AUTODFT_LOOP_INTERVAL` | `pipeline.loop_interval_seconds`          |
| `AUTODFT_ORCA_PATH`     | `orca.path`                               |
| `AUTODFT_ORCA_EXTRA`    | `orca.extra_args`                         |
| `AUTODFT_NBO_EXE`       | `orca.nbo_exe`                            |
| `AUTODFT_TMP_DIR`       | `orca.tmp_dir`                            |
| `AUTODFT_SESSION_SECRET` | `security.session_secret`                |

---

## Initialise

```bash
autodft admin init-db --config config/reaction.toml
```

This creates `data_path`, the `comp_data/` and `export_data/`
subdirectories, the SQLite database, all tables, **and seeds the six
standard ORCA headers** into the `computation_headers` table on the
first run:

| # | kind         | description                      |
| - | ------------ | -------------------------------- |
| 1 | confsearch   | GOAT GFN2-xTB conformer ensemble |
| 2 | confsearch   | GOAT g-xTB conformer ensemble    |
| 3 | singlepoint  | wB97X-D3 / def2-QZVPD            |
| 4 | optimization | wB97X-D3 / def2-SVP              |
| 5 | optimization | wB97X-D3 / def2-TZVP             |
| 6 | singlepoint  | wB97X-D3 / def2-TZVP             |

All six are method only: functional and basis (GOAT and its xTB level
for the conformer searches), without `%pal` / `%maxcore`. Grid,
convergence, RI, solvation, cores and memory are the `advanced`
settings below, whose defaults reproduce the full headers of before
Advanced settings (DEFGRID3, TightSCF, RIJCOSX def2/J, CPCM(MeCN)). A
submission that sets neither `header_*` nor `header_*_id` gets 2, 4 and
6; 5 and 3 are the same method on the larger def2-TZVP/def2-QZVPD basis
pair, kept as a seeded alternative. The defaults request 12 cores × 2000 MB
(optimisation) and 8 cores × 1500 MB (singlepoint), below the CPU
nodes' ~2600 MB/core. Measured on a 45-atom acridine: opt+Freq is 8 %
faster at 2000 than at 1250 MB, UV/Vis and SOC TDDFT are unchanged
between 1000 and 2500 MB, and singlepoints peak at 2–8 GB. The
double-hybrid NMR option runs on its own cores and memory
(`nmr_dh_nprocs` × `nmr_dh_maxcore`).

The headers can be edited / extended in the dashboard's **Headers** page
(full headers of before Advanced settings carry a "legacy" badge and are
not offered on the submit form) or via the `/api/headers` endpoints.
Anyone may use them; only admin creates, edits or deletes a header,
because its text goes into the ORCA input verbatim. A header in use
keeps its text and kind: an edit of either is saved as a new header.
When a submission form or CLI/API call leaves a header slot on its
default, the dashboard's header dropdown preselects the stored row
whose text equals the package default, so what you see selected is what
will actually run.

The same first run also **creates the `admin` account and logs its API
key once**, in a banner. That key is stored only as a hash, so copy it
out of the log — if you miss it, run
`autodft admin rotate-key admin --config config/reaction.toml`.

After upgrading to a build that stores job results, run
`autodft admin backfill-results --config config/reaction.toml [--project P]`
once to store records for jobs judged earlier, on an older parser version,
or without a matching identity. Readers fall back to the files without a
record, but they never save what they parse — so until the backfill runs,
every read of an old job re-parses its files again. Safe to run on the
controller host while the controller keeps going, and safe to repeat.

---

## Run the controller

```bash
autodft run --config config/reaction.toml
```

That starts the worker loop (one tick every `loop_interval_seconds`)
and, if `api.enabled = true`, boots the FastAPI dashboard at
`http://<host>:<port>/`. Press Ctrl-C to stop. The standard way to leave
it running on this cluster is inside a `screen` session. The log also
goes to `<data_path>/logs/controller.log` (rotated at 10 MB, five kept).
Production always starts it as
`.venv/bin/python -m autodft run --config config/reaction.toml` from the repo root.

Before any scheduler or database work, `run` takes `<data_path>/controller.lock`,
which records the host, pid and start time. A second `run` exits 1 and names
the holder while the lock names a live pid on this host or names another host;
the message says to delete the lock if no controller is running. A lock left on
this host by a dead pid (a killed controller) is taken over, and one naming this
host and the process's own pid (an NFS `link()` retransmit) counts as taken. The
lock is removed on exit. To move the controller to another host, stop it
there first, or delete a lock whose controller is gone for certain.

For local testing without SLURM:

```bash
autodft run --config test.toml --scheduler local --once   # single tick, then exit
```

Never run `--once` against the live controller's database: two submitters
defeat the `SUBMITTING` claim. The controller lock refuses it while the
controller runs.

---

## Authentication

Every route except `/login` and `/logout` needs a credential. There is
exactly one, and it names a person:

* **API key** — `X-AutoDFT-API-Key: adft_…` (or
  `Authorization: Bearer adft_…`) identifies the key's owner. Admin
  included: there is no shared password, because a secret everyone knows
  authenticates a crowd rather than a caller.
* **Session cookie** — `autodft_auth`, set by `/login`, where a browser
  exchanges **username + API key** for it once. 7-day default lifetime;
  `/logout` clears it. The cookie is bound to the key it was issued
  under: rotating that key signs the user out everywhere. Cookies from
  before this binding are refused too, so everyone signs in once again
  after the upgrade.

Lost a key — the admin key included? `autodft admin rotate-key <user> --config config/reaction.toml`
issues a new one. It needs a shell on the controller, which is the point:
recovery is local rather than a second, weaker credential.

```bash
# Browser flow — visit http://localhost:8085/ and you'll be redirected
# to /login.

# Script flow — header on every request:
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..." | jq .

# Which account am I, and what do I own?
curl -s http://localhost:8085/api/whoami \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..." | jq .
```

Keys are minted by admin — Admin → Users in the dashboard, or
`POST /api/admin/users` — and shown **once**, at creation or rotation.
Unauthenticated `/api/*` requests get 401; browser requests get a 303 to
`/login`. The example scripts under `examples/` read their credential
from a constant at the top of the file (`AUTODFT_API_KEY`) — set it to
match your deployment.

Field-level detail on credentials, namespaces and account management is
in [`docs/API.md`](docs/API.md) §0.

---

## Dashboard

The single-page UI at `http://<host>:8085/` has a left sidebar with five
pages. Everything on them is filtered to the signed-in account; admin
sees the whole database:

* **Job Submission** — the submission form. SMILES (validated live —
  RDKit error or canonical / atom-count / charge / mult shown inline
  before you can submit), project, priority, request flags
  (Skip-confsearch / T1 / ox / red / Vertical excitations), per-state
  conformer-count inputs that appear only for the states you've
  requested (default **1 per state**), three kind-filtered header
  dropdowns (legacy full headers left out) and a collapsed **Advanced**
  section: grid, SCF / geometry convergence, RI, auxiliary basis,
  solvation, cores and memory per stage plus the imaginary-frequency
  tolerance, pre-filled from the config defaults. The Submit button is
  disabled while the SMILES isn't valid; the server re-validates anyway,
  so direct API users also get HTTP 400 on bad input. The Author field
  is pre-filled with your username and is read-only unless you are
  admin, and the project you type lands in your own namespace.
* **Current Status** — five live stat cards (Queued / Pending /
  Running / Failed / Molecules), three tables (queued entrypoints,
  SLURM-pending jobs, currently running jobs), and a **Failed
  Entrypoints** widget for SMILES the controller couldn't expand into
  tasks.
* **Project Overview** — pick a project from the dropdown to see its
  molecules, submission progress, success rate, and per-row task counts.
  Subpages: **Summary**, **Molecules** (per-conformer status, conformers
  numbered as in the exports with an unsuccessful optimisation shown as "—",
  with a rendered structure, drawn by RDKit.js 2026.9.1 from unpkg, pinned
  with SRI and fetched on the first structure shown; the SMILES text stays
  if it cannot load; three tickboxes, unticked on every project load: **Show
  energies** puts E (Eh) under every successful singlepoint but NMR/UV-Vis
  and G = E + G−E(el) of the parent opt (raw, no quasi-RRHO) under the
  relaxed one, **Show densities** adds ElDens/SpinDens buttons that open a
  3Dmol pop-up, **Show downloads** adds `.out` buttons to successful opts
  and singlepoints and `.xyz` to opts), **State Analysis** (triplet / redox
  energies, E vs SCE only when the opt and singlepoint of both states ran in
  MeCN, otherwise ΔG and a note; downloadable as XLSX), **Photophysics** and
  **Methods** (read-only: headers, every setting and its source, cutoffs,
  the full ORCA input of every job type and the pipeline settings). The
  Export panel has three buttons: **Export Project**, **Export and Delete**
  and **Delete Project** (see below).
* **Headers** — list computation headers; admin creates, edits and
  soft-deletes them (other users see the page read-only). Each
  has a `kind` (confsearch / optimization / singlepoint / any), a
  free-form description (shown in the submission dropdowns), and the
  raw ORCA block. Multi-line headers including blocks like
  `%xtb XTBInputString "--gxtb" end`, `%cpcm SMD true SMDsolvent "water" end`,
  or `%pal nprocs N end` are supported verbatim. Headers referenced
  by `created` or `pending` tasks can't be deleted; finished
  (successful / failed) references don't block the delete. Anyone
  may use a header; creating, editing or deleting one answers 403 to
  anyone but admin, and headers users made earlier stay usable. A
  deleted header is revoked for users: a new submission naming it (by
  ID or text) answers 400, and so does Extend of a project that uses it;
  what already runs on it keeps running.
  Editing the text or kind of a header any state or task uses saves a
  new header ("In use; saved as new header #N") and leaves the old one
  as it ran; a description edit stays in place.
* **Admin** — wipe one of your projects or a single molecule (each
  shows a preview and wants the exact name typed back), and, for admin
  only, the **Users** section (create an account, rotate a key,
  deactivate), the submission hold, the per-owner failure circuit breaker
  (each tripped owner listed with its own reset button) and the
  database reset.

**Export Project / Export and Delete / Delete Project** — the project page's
three buttons:

| Button | Writes the export | Deletes `comp_data` | DB rows | Afterwards |
|---|---|---|---|---|
| **Export Project** | yes | no | kept | live |
| **Export and Delete** | yes | yes | kept, molecules archived | read-only history |
| **Delete Project** | no | yes | deleted | gone |

The two export buttons open one dialog: **Lowest conformer per state** or
**All conformers**, and **Keep .gbw files**. Both write the same tree into
`<export_data>/<owner>/<project>/`, replacing a previous export there:
`<project>.csv` (reported conformer per state), `<project>_all_conformers.csv`,
`<project>_state_analysis.xlsx`, the photophysics workbook for a project with
photophysics molecules, the reactivity workbook (`_reactivity.xlsx` and
`.json`: sheets `molecular`, `atomic`, `wiberg`, one row set per relaxed and
vertical singlepoint of the conformers chosen in the dialog) for a project
with Reactivity molecules, and `molecules/<mol_id>/` with a `molecule.json`
(the mapping back to the database) and `<state>/conf<N>/opt.inp opt.out
opt.xyz sp.inp sp.out vert_ox.* uvvis.* …`. A molecule holding two states
of one name (resubmitted at other headers, or the NMR reference project's S0
per NMR method) writes them to `<state>__state<state_id>/`, and then both
CSVs add `state_id` and the three header ids to every row. The state analysis
has one row per method family (a molecule's states at one header set). Only
the final files of successful tasks are copied; a conformer with an imaginary
mode gets no folder and is named in the all-conformers CSV. **Export and
Delete** then deletes every `<comp_data>/mol_*/` of the project — everything
not in the tree is gone — flagging each molecule `archived` just before its
folder is deleted; the project stays browsable in Project Overview but can no
longer be exported (409). Refused for a protected project and for one that is
already archived; the dialog asks for the project name to be typed back. From
its start the project gets no new job or `sbatch`, and it waits until every
job the project sent to SLURM has finished and been judged, so a result that
lands during the wait is exported, not deleted; a job whose `sbatch` reported
failure is first looked up in SLURM and, if found, waited for too, and so is
a cube build or `.out`/`.xyz` download of the project already under way. An
archive cut short (a restart, an NFS error) can be re-run: it keeps the
export folders of the molecules already archived, deletes the raw folders
they have left, and exports and deletes the rest. A molecule whose raw data
is gone but which is not flagged (an archive cut short under older code)
keeps its export folder and is flagged. The controller deletes an archived
molecule's raw folder left behind at its next start, if the project's export
folder exists and one of the molecule's job paths lies in that raw folder. A
plain Export Project of such a project keeps those folders too, and so leaves
a molecule whose raw data is gone as it is. While it runs, a molecule wipe,
Delete Project, density cube builds, `.out`/`.xyz` downloads and another
export or archive of the project answer 409 ("Export and Delete is running
for this project."); they work again once it ends, failed or not.
**Delete Project** is the project wipe (below) offered to the owner and
admin: a preview, the project name typed back, refused for a protected
project and with 409 while another wipe runs or the project has an export
or archive running; until its rows are gone, an export or archive of the
project answers 409. "Also delete exported data" starts unticked. The Admin
page keeps its own wipe.

**"Add calculations…" (Extend)** — on the project page, before the project
is archived. It opens the submission form with the project's molecules and
headers: what every molecule already has is ticked and greyed out, and
anything else can be ticked — T1/ox/red, UV/Vis, IR, NMR, ESD with any
Herzberg–Teller mode. Each molecule gains only what it lacks; nothing is
recomputed. A large library can so run its ground states first and its
spectroscopy or ESD later. Densities, NBO and Reactivity run inside the energy
singlepoints and can only be chosen at the first submission; ESD needs the
S0 Hessians, which every optimisation keeps.

Refresh poll is every 5 s; the sidebar footer shows connectivity and
last-refresh time.
The project pages keep the project list and Summary they loaded for up to
30 s: switching between them within that time refetches neither. The refresh
button and an action that changes projects (submit, extend, export, Export
and Delete, a wipe, a database reset) load both afresh; choosing another
project loads its Summary.

A page load or user action that takes longer than 250 ms runs under a
page-blocking loading overlay with real progress; Cancel appears after 60 s.
Polling never shows it.

---

## Submit work

Every submission path (CLI, REST, Python) ends up writing the same
`CalculationEntrypoint` row. The full option matrix is documented in
[`docs/API.md`](docs/API.md#post-apisubmit); the highlights:

* `request_t1 / request_ox / request_red` — extra states beyond S0.
* `skip_confsearch` — skip GOAT, use RDKit's initial geometry for
  optimization directly.
* `seed_from_s0` — T1 / ox / red run no conformer search of their own; once
  every S0 conformer is finished, each optimises from the lowest S0 conformer.
* `request_singlepoint_vertical_excitations` (default **true**) —
  vert-ox / vert-red / spin-flip singlepoints on each optimised state.
* `max_conformers_S0 / _T1 / _ox / _red` — per-state conformer cap
  (default **1 per state**). Legacy `max_conformers` still works as a
  blanket override.
* `request_spec_uvvis` / `request_spec_ir` / `request_spec_nmr` / `request_esd`
  — UV/Vis (TDDFT singlepoint on every S0 conformer), IR (from the
  optimisation's frequencies), NMR (¹H/¹³C/¹⁹F shifts referenced to chloroform
  (default, its shift in the calculation's solvent added) or TMS
  (`nmr_reference` / `--nmr-reference`) and CFCl₃, closed-shell molecules only; `nmr_method` / `--nmr-method`
  `singlepoint` (default, the singlepoint header) or `dh`
  (revDSD-PBEP86-D4/2021 / pcSseg-3, H–Kr; own cores × memory
  `nmr_dh_nprocs` × `nmr_dh_maxcore`, default 12 × 6000 MB); computed in the protected
  `admin/reference_systems` project — the bare name `reference_systems`
  is reserved for every owner, not just admin), and
  ESD (S1/T1 excited-state dynamics — ISC, RISC, fluorescence,
  phosphorescence, IC only with `esd_ic` / `--esd-ic` — via ORCA's ESD module, closed-shell molecules only,
  much more expensive: a numerical S1 Hessian; TDA throughout except IC,
  TightOpt for its optimisations, Herzberg–Teller only where the SOC is
  weak, FC reruns flagging width- and soft-mode-sensitive ISC rates;
  S1→Tn window 0.3 eV by default). Only added to new
  molecules; see [`docs/API.md`](docs/API.md#post-apisubmit). See
  [docs/PHOTOPHYSICS.md](docs/PHOTOPHYSICS.md) for the methods, deployment
  and rollback.
* `request_densities` (CLI `--densities`, plus `--no-eldens` /
  `--no-spindens` / `--density-grid` / `--eldens-file` / `--spindens-file`)
  — Gaussian cube files (electron and/or spin density) from a `%plots`
  block added to every state's own energy singlepoint; spin density only
  where the state is open-shell. `request_singlepoint_nbo` (CLI `--nbo`,
  plus `--nbo-keywords`) — natural population analysis via a `%nbo` block
  on the same singlepoint, using ORCA's NBO 7. Both need the energy
  singlepoint stage and apply to **every** state's own singlepoint (S0,
  T1, ox, red — not ESD's S1, which has none), not only S0's. NBO needs
  `[orca].nbo_exe` configured; a submission is refused otherwise. See
  [docs/PHOTOPHYSICS.md](docs/PHOTOPHYSICS.md#densities-and-nbo).
* `request_sp_reactivity` (CLI `--reactivity`) — Hirshfeld, `%elprop`
  polarizability/quadrupole, NBO with Wiberg indices and a kept `.gbw` in
  every relaxed and vertical singlepoint, plus stability analysis on
  open-shell verticals (~350–450 GB of `.gbw` per 1,000 molecules). Needs
  `[orca].nbo_exe` and a singlepoint header without Hirshfeld, NBO,
  `%elprop`, `%nbo` or `STABPerform`; submission only, not Extend.
  Export Project writes the reactivity workbook.
* `imag_freq_tolerance` (CLI `--imag-freq-tolerance`, default **0**,
  −300 to 0) — every optimisation, ESD's S1 included, fails on any
  imaginary mode below it (cm⁻¹) and is retried; exports leave such
  conformers out. `-10` accepts noise modes down to −10 cm⁻¹.
* `advanced` (CLI `--opt-grid`, `--sp-grid`, `--opt-scf-convergence`,
  `--sp-scf-convergence`, `--opt-geom-convergence`, `--opt-ri` / `--sp-ri`,
  `--opt-aux-basis` / `--sp-aux-basis`, `--opt-solvation` / `--sp-solvation`,
  `--opt-solvent` / `--sp-solvent`, `--cs-nprocs` / `--opt-nprocs` /
  `--sp-nprocs`, `--cs-maxcore` / `--opt-maxcore` / `--sp-maxcore`) — grid,
  SCF and geometry convergence, RI and auxiliary basis, solvation, cores
  and memory per stage, rendered into every job; missing ones take the
  config `[advanced.*]` defaults (`GET /api/advanced-defaults`). Headers
  then hold only the method: one with e.g. `DEFGRID3`, `TightSCF`,
  `CPCM(...)`, `Opt`, `Freq`, `KeepDens`, `%pal` or `%maxcore` is refused,
  naming the setting. Without `advanced`, a submission to a project whose
  molecules ran on exactly these (full) headers stays as it was. An existing
  molecule is not resubmitted at the same headers with other method settings
  (cores and memory may differ); Extend takes the molecule's own; see
  [`docs/API.md`](docs/API.md#post-apisubmit).
* `header_confsearch_id / _optimization_id / _singlepoint_id` — pick a
  stored header by ID. Or pass raw `header_*` text; from anyone but
  admin it must be a stored header's or a default's text (403 otherwise),
  and a deleted header answers 400 by ID or text.
  The method-only defaults from `autodft/qm/orca/defaults.py` apply when
  neither is set.

`request_S1` is not exposed — S1 isn't supported yet.

The `project` you send is a **bare** name and lands in your own
namespace: `screening` submitted by `alice` is stored as
`alice/screening`. `author` is always your username — the field is
accepted and ignored, admin included.

### Backpressure and priority

Submission never blocks on the cluster. Every request is accepted the
moment it arrives and parked in `calculation_entrypoints`, so a script can
hand over a library of any size in one pass without hanging — that table
*is* the buffer. Throttling happens downstream, in two places:

* **Jobs → SLURM.** The controller keeps submitting until `squeue` reports
  `priority * queue_slots_per_priority` of **our own jobs waiting for the
  cluster** — jobs slurmctld has evaluated and cannot start yet. A job is
  `PD` from the moment `sbatch` returns, because SLURM schedules on its own
  cycle rather than on submit, so jobs whose pending *reason* is still
  `None` are not counted: doing so made the loop measure its own
  submissions and stop after one capful on an idle partition. Otherwise
  the cap works as before (default 10 per unit of priority, so `priority = 1` allows 10
  queued jobs, `priority = 5` allows 50). Running jobs never count, so on
  an idle partition submission continues until the cluster is full and
  jobs finally begin to queue. The queue depth is re-read from `squeue`
  every few submissions rather than assumed, because a job SLURM starts
  immediately is running, not waiting. Jobs are ordered by priority, so
  higher-priority work claims the slots first.

  One tick yields after `max_submission_seconds_per_tick` so a long
  submission run cannot starve status polling; it resumes on the next
  tick. `max_submissions_per_tick` is an optional count backstop, off by
  default — when it was on it became the binding constraint and capped
  the fill rate at one backstop-full per tick.
* **Entrypoints → jobs.** Expansion pauses once `max_unsubmitted_jobs`
  jobs exist in the database but have not reached SLURM, which keeps the
  on-disk `comp_data/` tree growing in step with what the cluster can
  actually absorb.

A molecule inherits its priority from the entrypoint that created it.
Resubmitting the same molecule at a higher priority raises it; a lower one
never demotes work already in flight.

Only the admin account sets a priority (1 to 1000); the API answers 400 to any
other account that sends one other than the default 10, and the dashboard hides
the field.

### CLI

The CLI writes to the database directly, so it has no API key to
identify itself: `--user` names the account whose namespace the project
lands in, and defaults to `admin`.

```bash
# minimal — defaults all the way (project becomes admin/alcohols)
autodft submit submit --config config/reaction.toml --smiles CCO --project alcohols

# submit on someone's behalf — project becomes alice/phenols
autodft submit submit --config config/reaction.toml --smiles CCO --project phenols --user alice

# full coverage of all options
autodft submit submit --config config/reaction.toml \
    --smiles "c1ccc(O)cc1" \
    --project phenols \
    --priority 20 \
    --request-t1 --request-ox --request-red \
    --no-vert-ex \
    --max-conformers-s0 5 \
    --max-conformers-t1 3 \
    --max-conformers-ox 2 \
    --max-conformers-red 2 \
    --header-confsearch path/to/cs_header.txt \
    --header-opt        path/to/opt_header.txt \
    --header-sp         path/to/sp_header.txt \
    --sp-grid DEFGRID2 --opt-nprocs 16 --sp-solvation SMD --sp-solvent water

# batch from CSV — same flags accepted, same defaults
autodft submit submit-batch --config config/reaction.toml --file batch.csv --project phenols --priority 20

# skip-confsearch path (RDKit geometry → optimization directly)
autodft submit submit --config config/reaction.toml --smiles CC --project quick --skip-confsearch
```

### REST API

The controller exposes a JSON API on the same port as the dashboard.
Every call carries an API key. The shortest path:

```bash
export AUTODFT_API_KEY=adft_7Kq2XnR4...

curl -X POST http://localhost:8085/api/submit \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles": "CCO", "project": "alcohols"}'
```

A full body using every option:

```bash
curl -X POST http://localhost:8085/api/submit \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{
           "smiles": "c1ccc(O)cc1",
           "project": "phenols",
           "priority": 20,

           "request_t1":  true,
           "request_ox":  true,
           "request_red": true,
           "skip_confsearch": false,
           "request_singlepoint_vertical_excitations": false,

           "max_conformers_S0": 5,
           "max_conformers_T1": 3,
           "max_conformers_ox": 2,
           "max_conformers_red": 2,

           "header_confsearch_id":   2,
           "header_optimization_id": 3,
           "header_singlepoint_id":  4
         }'
```

For anything library-sized use `POST /api/submit-batch`, which takes the
same options with `smiles` replaced by `smiles_list`. One request, one
transaction, and rejections are reported per SMILES instead of failing the
whole call:

```bash
curl -X POST http://localhost:8085/api/submit-batch \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles_list": ["CCO", "c1ccccc1"], "project": "alcohols"}'
```

The endpoint returns `400` with the RDKit reason if the SMILES is
invalid — no row is queued. Submit and submit-batch also answer `400` for a
project name that is not bare (`/` or `:` in it, or anything but 1–64
letters, digits, `.`, `_`, `-`) and for a negative `max_conformers*` (Extend
too; `0` asks for none). Submit-batch and Extend reject an entry over 512
characters on its own, in `rejected`. Validate a SMILES without submitting via
`POST /api/validate-smiles`, or up to 100 in order via
`POST /api/validate-smiles-batch` (`{"smiles_list": [...]}`; a line over 512
characters comes back invalid, more than 100 lines answer 422):

```bash
curl -X POST http://localhost:8085/api/validate-smiles \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' -d '{"smiles":"xxx"}'
# -> {"valid": false, "error": "RDKit could not parse 'xxx'.", ...}
```

Full route list. `{name}` is a project, written `owner:project` in a URL
(a bare name means "mine"). `/api/admin/*` is admin-only; everything else
is either scoped to the caller's own projects or deliberately shared
(the headers library, cluster status, SMILES validation):

| Method | Path                                | Purpose                                                       |
| ------ | ----------------------------------- | ------------------------------------------------------------- |
| GET    | `/`                                 | HTML dashboard (SPA)                                          |
| GET/POST | `/login`                          | sign-in form and its submission — no credential needed        |
| GET    | `/logout`                           | clear the session cookie — no credential needed               |
| GET    | `/api/whoami`                       | the signed-in account and the projects it owns                |
| GET    | `/api/cluster`                      | read-only queue depth + the caller's breaker state (admin: any owner's), for every account |
| GET    | `/api/overview`                     | counts of molecules / tasks / jobs / queue                    |
| GET    | `/api/molecules`                    | list molecules (`?project=&limit=&offset=`)                   |
| GET    | `/api/molecules/{id}`               | full molecule with states / tasks / jobs                      |
| GET    | `/api/tasks`                        | list tasks (`?status=&type=&limit=`); 400 for an unknown status or type |
| GET    | `/api/jobs`                         | newest jobs (`?status=&limit=`): a SLURM state or the group `running`/`pending`; `X-Total-Count` counts all matches |
| GET    | `/api/queue`                        | unstarted entrypoints                                         |
| GET    | `/api/entrypoints/failed`           | entrypoints whose processing raised before any task was made  |
| GET    | `/api/headers`                      | seeded defaults + custom headers (`?kind=…&include_deleted=`) |
| POST   | `/api/headers`                      | create a custom header (admin; 403 otherwise)                 |
| PUT    | `/api/headers/{id}`                 | update text / description / kind / validated (admin)         |
| DELETE | `/api/headers/{id}`                 | soft-delete (admin; blocked by in-flight tasks)               |
| POST   | `/api/validate-smiles`              | validate a SMILES string                                      |
| POST   | `/api/validate-smiles-batch`        | validate up to 100 SMILES in order (used by the live form)    |
| POST   | `/api/submit`                       | submit a new SMILES                                           |
| POST   | `/api/submit-batch`                 | submit many SMILES in one request (see below)                 |
| GET    | `/api/projects`                     | list projects with summary counts                             |
| GET    | `/api/projects/{name}`              | per-project molecules + progress + success rate               |
| GET    | `/api/projects/{name}/molecules-detail` | per-conformer status for every molecule                   |
| GET    | `/api/projects/{name}/sp-energies`  | E of successful singlepoints, G on relaxed ones               |
| GET    | `/api/projects/{name}/sp-densities` | which singlepoints have ElDens/SpinDens cubes to show         |
| POST   | `/api/projects/{name}/densities/{task_id}/{kind}` | one cube, built with orca_plot if needed (503 while busy) |
| GET    | `/api/projects/{name}/tasks/{task_id}/files/{out\|xyz}` | a successful opt's/SP's `.out`, an opt's final `.xyz` |
| GET    | `/api/projects/{name}/state-analysis` | triplet / redox / reorganisation energies                   |
| POST   | `/api/projects/{name}/state-analysis/export` | the same, as a multi-sheet XLSX (background job)    |
| GET    | `/api/projects/{name}/photophysics` | UV/Vis, IR, NMR and ESD results per molecule (`?molecule_id=`) |
| GET    | `/api/projects/{name}/methods`      | headers, settings, cutoffs, ORCA inputs, pipeline config      |
| GET    | `/api/projects/{name}/extend-info`  | what the project's molecules already have (for Extend)        |
| POST   | `/api/projects/{name}/extend`       | add categories / states to the project's molecules            |
| POST   | `/api/projects/{name}/export`       | trigger CSV/JSON/files/photophysics/reactivity export (`?format=&all_conformers=&keep_gbw=`) |
| POST   | `/api/projects/{name}/archive`      | **destructive**: the export tree, then wipe `comp_data`        |
| GET    | `/api/projects/{name}/jobs`         | the project's background jobs (exports, archive), newest first |
| GET    | `/api/jobs/{id}`                    | one background job                                            |
| GET    | `/api/jobs/{id}/download`           | the file a finished export produced                           |
| GET    | `/api/projects/{name}/wipe-preview` | what a project wipe would delete — counts only                |
| POST   | `/api/projects/{name}/wipe`         | **destructive**: a project's rows, `comp_data`, exports        |
| GET    | `/api/molecules/{id}/wipe-preview`  | what a molecule wipe would delete — counts only               |
| POST   | `/api/molecules/{id}/wipe`          | **destructive**: one molecule's rows and files                 |
| GET    | `/api/wipe-status`                  | progress of a deletion still running in the background        |
| GET    | `/api/admin/circuit-breaker`        | per-owner breaker states with the failure ratios behind them (admin) |
| POST   | `/api/admin/circuit-breaker/reset`  | clear one owner's breaker (body `{"owner": …}`) or all, and resume their submissions (admin) |
| GET    | `/api/admin/submission-hold`        | whether submission is paused by hand (admin)                  |
| POST   | `/api/admin/submission-hold`        | pause `sbatch` submission (`{"reason": …}`); results keep being processed (admin) |
| POST   | `/api/admin/submission-hold/release`| resume submission on the next tick (admin)                    |
| GET    | `/api/admin/reset-preview`          | everything a database reset would delete — row counts, no files |
| POST   | `/api/admin/reset-database`         | **destructive**: every pipeline table and data directory       |
| GET    | `/api/admin/disk-usage`             | the last disk measurement, or `null` — never starts one (admin) |
| POST   | `/api/admin/disk-usage`             | measure the data directory on a background thread (admin)     |
| GET    | `/api/admin/users`                  | list accounts                                                 |
| POST   | `/api/admin/users`                  | create an account; the response carries its API key, once     |
| POST   | `/api/admin/users/{name}`           | deactivate / reactivate (`?active=false`)                     |
| POST   | `/api/admin/users/{name}/rotate-key`| new key; the old one and its sessions stop working immediately |
| POST   | `/api/admin/projects/{name}/reassign`| move a project, its job rows and export directory to another owner (409 while a job runs or if the target exists) |

### Accounts

One **admin** account reaches everything. Every other account is a
**user**: one API key, their own projects, nothing else. Users sign in
with username + API key, submit with the key in `X-AutoDFT-API-Key`, and
see only their own work. The `author` on a submission is always the
calling account's username and is not editable by anyone, admin included.

Projects are namespaced per owner — stored as `owner/project`, written
`owner:project` in a URL, and submitted as a bare name that lands in the
caller's namespace. Two people can each have a `screening`.

Every caller presents a key that names them. Full detail, including account
management, is in [`docs/API.md`](docs/API.md) §0.

End to end, as a new user (`$K` is the key admin handed you):

```bash
K="X-AutoDFT-API-Key: adft_7Kq2XnR4..."
H="http://localhost:8085"

# 1. who am I?
curl -s -H "$K" $H/api/whoami            # -> {"username":"alice", ...}

# 2. submit — the bare project name lands in your namespace
curl -s -X POST -H "$K" -H 'Content-Type: application/json' \
     -d '{"smiles":"CCO","project":"screening"}' $H/api/submit

# 3. find it — the listing shows the qualified name. It appears once the
#    controller has expanded the entrypoint into a molecule; until then
#    the project is only in /api/whoami.
curl -s -H "$K" $H/api/projects          # -> [{"name":"alice/screening", ...}]

# 4. read it — ':' replaces '/' in the URL (a bare name means yours)
curl -s -H "$K" "$H/api/projects/alice:screening"

# 5. export it — writes <export_data>/alice/screening/screening.csv
curl -s -X POST -H "$K" \
     "$H/api/projects/alice:screening/export?format=csv"
```

Every wipe and the database reset require an exact confirmation string
(the qualified project name, the molecule's SMILES, or `RESET THE
DATABASE`), refuse to run while another one is in flight (409), while
the project (any project, for the reset) has an export or archive running
(409), or while one of its jobs is mid-`sbatch` with no SLURM id to cancel
yet (409). That clears within a tick, but a claim a crash left `SUBMITTING`,
or a job whose `sbatch` reported failure, blocks the wipe of its project,
the wipe of its molecule and the reset until the controller has run again
and resolved it. The rows go first; the jobs SLURM still has are then
`scancel`ed (given up after 60 s, which the response's `scancel_warning`
names), and only then the files are renamed aside and unlinked on a
background thread, so the request returns in well under a second while a
project-sized deletion runs for minutes — poll
`GET /api/wipe-status` for that. `POST /api/projects/{name}/archive` takes the
same `confirm` (the qualified project name; 400 without it), and the Export
and Delete dialog asks for the name to be typed back.

Field-level reference for each endpoint, including every option of
`POST /api/submit`, is in [`docs/API.md`](docs/API.md).

### Python

For embedded use, write directly to the database from Python — see
`examples/01_submit_via_python.py`. The package exposes:

```python
from autodft.db import get_session, init_db
from autodft.engine.entrypoint_processor import validate_smiles
from autodft.models.entrypoint import CalculationEntrypoint
from autodft.qm.orca.defaults import (
    DEFAULT_HEADER_CONFSEARCH,        # GOAT g-xTB
    DEFAULT_HEADER_OPTIMIZATION,      # wB97X-D3 / def2-SVP, method only
    DEFAULT_HEADER_SINGLEPOINT,       # wB97X-D3 / def2-TZVP, method only
    GFN2_HEADER_CONFSEARCH,           # GOAT GFN2-xTB alternate
    QZVPD_HEADER_SINGLEPOINT,         # larger-basis alternate
)
from autodft.qm.orca.advanced import for_submission  # resolves `advanced`
```

An entrypoint inserted without `advanced` runs its headers as they are,
so the controller refuses one whose optimization header is method only;
`for_submission` fills `advanced` from the config defaults.

---

## Monitor

The CLI talks to the database directly, with no account behind it, so
`--project` wants the **qualified** name (`admin/phenols`, not
`phenols`) — that is the string stored on every molecule. Each command
reads the database of `--config`, by default the checkout's
`config/reaction.toml`. They, and every `submit` and `admin` command, exit 1,
naming the host, when `<data_path>/controller.lock` says the controller runs on
another host: the SQLite database on NFS may be opened from one host only.
Without a lock file they proceed:

```bash
autodft status overview                          # totals
autodft status queue                             # waiting entrypoints
autodft status molecules --project admin/phenols
autodft status molecule 42                       # one molecule's full tree
autodft status tasks --status pending
autodft status jobs --status RUNNING
autodft admin progress --project admin/phenols   # submission / success rate
```

Or hit the same data over HTTP (`/api/overview`, `/api/tasks?status=…`,
`/api/entrypoints/failed`, …) — see `examples/03_monitor_progress.py`.

---

## Export

Exports are written under `<export_data>/<owner>/<project>/`, and the
CLI takes the qualified project name.

Energy summary:

```bash
# CSV — default destination is <export_data>/admin/phenols/phenols.csv
autodft admin export --project admin/phenols --config config/reaction.toml

# JSON, all conformers, custom path
autodft admin export \
    --project admin/phenols --format json --all-conformers \
    --output /tmp/phenols.json --config config/reaction.toml
```

A CSV or JSON export with no results to write prints "Nothing to export" and
removes the old file at the target path; the API job answers the same, with no
download. The command refuses an archived project, whose kept files are the
only copy. It and `export-files` exit 1 while the project has an archive or
export job running: both write its export directory.

The export tree (summary files and `molecules/<id>/<state>/conf<N>/…`, as
"Export Project" writes it):

```bash
autodft admin export-files \
    --project admin/phenols --config config/reaction.toml [--all-conformers] [--keep-gbw]
# -> /mnt/share/dft_calculations/autodft_data/export_data/admin/phenols/
```

Or click **Project Overview → Export Project** in the dashboard, which calls
`POST /api/projects/{name}/export?format=files` and writes into the same
`<export_data>/<owner>/<project>/`.

Programmatically (see `examples/04_export_results.py`):

```python
from autodft.extraction.extractor import PipelineExtractor

# The qualified name, exactly as it is stored on the molecules.
ext = PipelineExtractor("admin/phenols")
ext.export_summary_csv("phenols.csv")
ext.export_tree("./export_data", all_conformers=False, keep_gbw=False)  # -> ./export_data/admin/phenols/
```

---

## Failure handling

The pipeline is built to **fail loudly**, not silently:

* SMILES that RDKit/OpenBabel can't turn into a 3-D geometry mark their
  entrypoint with `processing_error` and appear on
  `/api/entrypoints/failed` and the dashboard's "Failed Entrypoints"
  widget. The controller never silently retries them.
* A task normally transitions to `failed` after `pipeline.max_attempts`
  unsuccessful jobs. Before that it remains `pending` while retries are
  still possible — that's what the "Pending" card counts. A job uses an
  attempt when:
  * ORCA's checks fail on its output, or SLURM ends it (`FAILED`,
    `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`, …);
  * 3 consecutive sbatch rejections of its script (an error not on the
    transient list) fail it as `sbatch rejected 3 times`. Transient errors
    (timeouts, slurmctld unreachable, munge/authentication, submit limits)
    are retried for free, and a tick in which every sbatch failed is taken
    as an outage and counts no rejection. A job is marked `SUBMITTING`
    before its sbatch; one a crash leaves so is looked up in SLURM on the
    next tick and adopted, or submitted if SLURM never got it;
  * it is in a SLURM state the pipeline does not know:
    `Unrecognised SLURM state <state>`;
  * neither sacct nor squeue knows its job id on two polls running:
    `SLURM has no record of the job`.
* A task fails at once, without using its remaining attempts, when:
  * its pending job is unsatisfiable: the squeue Reason says the request
    can never run (`PartitionConfig`, `PartitionTimeLimit`,
    `MaxMemPerLimit`, `BadConstraints`, `DependencyNeverSatisfied`). The
    job is scancelled and fails as `unsatisfiable: <Reason>`. If more than
    ten pending jobs show the same Reason in one tick, it is taken as a
    cluster-side change: nothing is cancelled and a critical is logged;
  * the same failure would repeat on every retry (`Excited Root`,
    `LibXC Needed`, `S1 Stays Bound`);
  * an NMR MP2-memory retry is already at its cap and would rerun the
    failed job unchanged (`MP2 response memory cap reached`);
  * building its job raised 10 consecutive times
    (`Job build failed 10 times: …`). A failed build costs no attempt;
  * its project is archived: the project's created and pending tasks fail
    as `project archived`, which the circuit breaker does not count.

A retry after an abnormal termination, timeout or SLURM kill runs on
`pipeline.retry.increased_nprocs` cores or the job's own, whichever is more,
and `increased_time_limit` unless the stage's is longer. It keeps the job's
`%maxcore`. A memory failure (`OUT_OF_MEMORY` or an ORCA out-of-memory
message) instead keeps the job's cores and time limit and raises `%maxcore`
to `increased_mem_per_core` (at most 2500 MB), or 1.5× the job's own when
that is already as high. `max_mem_per_job_mb` (unset: 400 GB) bounds that
rise, and the extra cores of other retries, never below the job's own.
No retry lowers a job's cores or `%maxcore`.
SLURM's `--mem` is cores × (`%maxcore` + 50 MB) on a first attempt. Every
retry keeps the previous job's margin over that (its `--mem` ÷ its cores ×
(`%maxcore` + 50)) and applies it to the new cores × (`%maxcore` + 50); an
`OUT_OF_MEMORY` retry multiplies the margin by 1.25 first (1.25×, 1.5625×, …).
NMR MP2-memory retries keep it too. `--mem` stays within the same ceiling
(400 GB for an MP2-memory retry).

An optimisation with an imaginary mode is retried with TightSCF and
`%geom Convergence Tight` from the seed geometry; from the third attempt on,
the last attempt's final geometry is displaced
`pipeline.optimization.displacement` Å (0.1) along its lowest imaginary mode.
An output with several frequency blocks is judged on the last.

Recovering:

```bash
# each with --config config/reaction.toml
autodft admin reset-task 17                          # failed task to 'created'
autodft admin reset-task 17 --force                  # re-run in any status
autodft admin requeue-failed --project admin/phenols # bulk requeue
autodft admin cleanup --days 30                      # purge old completed entrypoints
```

A requeued task gets a fresh set of `pipeline.max_attempts` attempts; its
earlier attempts stay in the job history but no longer count towards failing
it again, and a success among them no longer counts either. A job of the old
run that was built but never sent is marked `superseded by a reset`.

`reset-task` refuses a task that is not `failed`. With `--force` it takes any
status, e.g. an optimisation to redo with an imaginary mode, and deletes the
task's jobs and every task built on its results: its singlepoints, category
singlepoints and ESD rate tasks, recursively, with their jobs, results,
geometries and folders under `comp_data`. A reset ESD seed (S0*) also
deletes the S1/T1 work and seeds ESD again. Resetting any other S0
conformer's optimisation does not pick S0* again, even if the new energy
would make that conformer the lowest. Once the new job succeeds, the
follow-ups are created again from its geometry.

`reset-task` refuses, and `requeue-failed` skips, a task with a job that
SLURM may be running: pending or running, claimed for sbatch (`SUBMITTING`),
or left by an sbatch that reported failure but may have been accepted. Both
also wait for a job that finished in SLURM but has not been judged yet
(next tick). `--force` deletes such a finished job, but refuses when SLURM
may be running a job of any task it would delete. The check and the changes
run under one database write lock, so the controller cannot submit a job in
between. A `SUBMITTING` claim left by a crash blocks a reset or requeue of
its task, and an archive of its project, until the controller runs again and
releases it.

`reset-task` (with or without `--force`) refuses, and `requeue-failed` skips
and counts, a task of an archived project, which nothing can re-run, and of a
project with an archive or export job running.

For an entrypoint that failed before being expanded (e.g. bad SMILES),
fix the SMILES and resubmit; the original `processing_error` row stays
in the queue history for auditing.

---

## Examples

The `examples/` directory contains runnable scripts:

Each file is a normal Python module: configuration lives in named
constants at the top (server URL, project name, etc.), the work is
factored into importable helper functions, and an
`if __name__ == "__main__":` block runs a demo. Use them as starting
points for your own scripts.

| File                          | What it shows                                                    |
| ----------------------------- | ---------------------------------------------------------------- |
| `01_submit_via_python.py`     | `submit()`, `make_metadata()`, `header_by_description()`, `validate_smiles()`. Direct DB submission covering every `request_metadata` option. |
| `02_submit_via_rest_api.py`   | `submit()`, `validate_smiles()`, `list_headers()`, `overview()`, `queue()`, `failed_entrypoints()`. Stdlib-only HTTP client. |
| `03_monitor_progress.py`      | `snapshot_via_db()`, `snapshot_via_api()`, `watch()`. SQLModel and HTTP backends with the same return shape. |
| `04_export_results.py`        | `project_progress()`, `export_summary()`, `export_files()`, `archive()` — built on `PipelineExtractor`. |
| `05_export_via_rest_api.py`   | `list_projects()`, `inspect_project()`, `export_project()`, `archive_project()` — same operations over the REST API. |

Each script is documented and self-contained.

---

## Project layout

```
autodft/
├── config/
│   ├── default.toml      framework defaults (do not edit)
│   └── reaction.toml     production config (data_path + [orca].path)
├── autodft/
│   ├── api/              FastAPI routes + SPA dashboard
│   ├── cli/              Typer commands (submit / run / status / admin)
│   ├── engine/           pipeline loop + state machine + SLURM scheduler
│   ├── extraction/       PipelineExtractor (CSV / JSON / files)
│   ├── models/           SQLModel tables
│   ├── qm/orca/          ORCA input/output + seeded default headers
│   ├── accounts.py       users, API keys, project ownership, migration
│   ├── paths.py          project-name validation + safe export paths
│   ├── config.py         Settings dataclasses + TOML/env loader
│   └── db.py             SQLite engine + sessions + header seed
├── docs/                 API reference, accounts design, upgrade note
├── examples/             see table above
└── tests/                pytest suite
```
