# AutoDFT REST API

The FastAPI controller exposes a JSON API on the same port as the
dashboard (`api.port`, default `8085`). All endpoints return JSON
unless noted. Errors come back as HTTP 4xx/5xx with a JSON body
`{"detail": "...", ...}`.

For interactive exploration, the controller also serves the OpenAPI
schema at `GET /docs` and `GET /openapi.json` — both behind the same
authentication as everything else, so sign in first.

---

## 0. Accounts and authentication

There is one **admin** account and any number of **users**. Admin reaches
everything. A user reaches only their own projects — listing, reading,
exporting, submitting — and cannot reach `/api/admin/*` at all.

### Credentials

| Credential | Sent as | Resolves to |
|---|---|---|
| API key | `X-AutoDFT-API-Key: adft_…` or `Authorization: Bearer adft_…` | the key's owner |
| Session cookie | `autodft_auth`, set by `/login` | whoever signed in |

A key is `adft_` plus 32 random characters. It is shown **once**, when the
account is created or the key is rotated, and stored only as
`sha256(key)` — it can never be read back, only replaced. Rotating takes
effect immediately, and ends every session cookie issued under the old
key: the cookie carries a fingerprint of the key (an HMAC of its stored
hash under the session secret, never the key), and a stale one gets **401** (API) or a **303** to
`/login` (browser). Rotating one user's key leaves everyone else's
sessions alone. Cookies issued before this check existed carry no
fingerprint and are refused, so every user signs in once after the
upgrade.

The `admin` account is created on the controller's first boot and its key
is logged once, in a banner. Every other key comes from
`POST /api/admin/users` (or the dashboard's Admin → Users), which is the
only place it is ever shown.

There is no shared password. `X-AutoDFT-Password` was removed in 0.5.5:
it authenticated a crowd rather than a caller, it resolved to admin — so
every holder had the destructive routes — and it made `project_author`
unattributable. Scripts that still send it get **401**; swap the header
for `X-AutoDFT-API-Key`.

```bash
# A user, with their key
curl -s http://localhost:8085/api/overview \
     -H "X-AutoDFT-API-Key: adft_7Kq2XnR4..."

# Browser: username + API key
curl -i -c cookies.txt -X POST http://localhost:8085/login \
     -d 'username=mhoffmann&password=adft_7Kq2XnR4...&next=/'
curl -s -b cookies.txt http://localhost:8085/api/overview
```

Lost a key, admin's included? `autodft admin rotate-key <user> --config <config>` issues a
new one and prints it once. It is deliberately local — a shell on the
controller and write access to the database — rather than a second,
weaker credential reachable over the network.

`GET /api/whoami` answers `{"username", "is_admin", "projects"}`, where
`projects` lists the **bare** names you own — the owner is you, so the
prefix would be noise. Qualify them with your username to address them
elsewhere.

Unauthenticated `/api/*` requests get **401**; unauthenticated browser
requests get a **303** to `/login?next=<original-path>`. A key belonging
to a deactivated account stops working immediately, as does a session
cookie issued to it.

### Project namespaces

Every project belongs to someone and is stored as `owner/project`, so two
people can each have a `screening`. Three forms appear:

| Where | Form | Note |
|---|---|---|
| stored | `mhoffmann/screening` | in `molecules.project_name` |
| in a URL | `mhoffmann:screening` | `/api/projects/mhoffmann:screening` |
| in a submit body | `screening` | qualified with the caller's namespace |

The URL form uses `:` rather than `/` because `/api/projects/{name}`
matches a single path segment, a percent-encoded slash is normalised back
to a separator before routing, and a `/api/projects/{owner}/{name}` route
would collide with `/api/projects/{name}/export` for any project called
"export".

Submitting is unchanged: send the bare name and it lands in your own
namespace. Submitting to a name someone else owns creates *your* project
of that name rather than joining theirs.

The `author` field is your username and is not editable — admin included.
To record work as someone else, submit it with their key.

Reads outside your namespace answer **404**, not 403 — a 403 would
confirm the project exists.

### Managing accounts (admin only)

| Endpoint | Does |
|---|---|
| `GET /api/admin/users` | every account, with projects and molecule counts |
| `POST /api/admin/users` | create one; the response carries the key, once |
| `POST /api/admin/users/{username}/rotate-key` | new key, old one and its sessions dead immediately |
| `POST /api/admin/users/{username}?active=false` | deactivate |
| `POST /api/admin/projects/{name}/reassign` | move a project to another owner |

Reassign takes `{"owner": "<username>"}` and moves the molecules, queued
entrypoints, background-job rows and the export directory
`export_data/<owner>/<project>` (an archive's frozen CSVs and photophysics
payloads included) to the new name, so an archived project stays browsable
and exportable for its new owner. It answers **409** while the project has
an archive or export running, and when the new owner already has a project
or an export directory of that name. The directory is renamed before the
database commit and renamed back if the commit fails. An archive or export
requested under the old name while the reassign runs answers **404**.

There is no delete-user. An account whose projects still hold hundreds of
gigabytes should not disappear in one click: wipe or reassign the
projects first, then deactivate.

### Saved headers

Every signed-in account may list and use headers — a method library is
only useful shared. Creating, editing or deleting one is admin's alone
(**403** otherwise), because header text goes into the ORCA input verbatim;
headers users created earlier stay usable. For the same reason a raw
`header_*` text in a submission from anyone but admin must equal a stored
header's or a package default's text. A deleted header is revoked for
everyone but admin: a submission naming it by id or by text answers
**400**, and so does Extend of a project whose S0 states use it, while
states and tasks that already use it keep running. A header
any state or task uses keeps its text and kind for good: an edit of
either is saved as a new header (see `PUT /api/headers/{id}`).

The six seeded methods belong to `admin`, as does anything created
before accounts existed.

`GET /api/cluster` is readable by everyone and reports only
`breaker_tripped` (the caller's own breaker; for admin, whether any
owner's is tripped) and `queued_entrypoints`, so
a user can tell "my jobs are stuck" from "the pipeline is halted" without
asking an administrator.

---

## 1. Submission

### `POST /api/validate-smiles`

Run RDKit on a SMILES string and return a structured verdict. Used by
the dashboard for live form validation; equally useful from a script.

**Body:**

```json
{ "smiles": "c1ccc(O)cc1" }
```

**Response:**

```json
{
  "valid": true,
  "canonical": "Oc1ccccc1",
  "atoms": 13,
  "heavy_atoms": 7,
  "charge": 0,
  "multiplicity": 1,
  "error": null
}
```

`valid=false` populates `error` with the RDKit reason. Rejected cases
include empty strings, anything RDKit refuses to parse, and single-atom
species (GOAT needs ≥2 atoms).

### `POST /api/validate-smiles-batch`

The same check for up to 100 SMILES, answered as a list in the order sent;
the dashboard's live form checks a pasted list this way, 100 lines a request.
A line over 512 characters comes back `valid=false` with `error` "SMILES too
long (>512)." (the same fields as any refusal); more than 100 lines answer
422. Any signed-in account.

```json
{ "smiles_list": ["CCO", "not-a-smiles"] }
```

### `POST /api/submit`

Queue one calculation entrypoint. The server validates the SMILES first
(same validator as above) and returns `400` if invalid. Resolves any
`header_*_id` against the `computation_headers` table; if neither
`header_*_id` nor `header_*` (raw text) is provided, falls back to the
package defaults in `autodft/qm/orca/defaults.py`: method only
(`!wB97X-D3 def2-SVP`, `!wB97X-D3 def2-TZVP`, GOAT g-xTB without `%pal` /
`%maxcore`), everything else comes from `advanced`. From anyone but
admin a raw `header_*` text must equal a stored header's or a package
default's text; any other text answers **403** (only admin adds header
text), and a soft-deleted header, by `header_*_id` or by its text, answers
**400**; on `POST /api/submit-batch` too. The controller refuses a deleted
header's text again when it expands the entrypoint of a non-admin author
(package defaults excepted), and a non-admin Extend whose project headers
include a deleted one.

**Body — every option:**

| Field                                       | Type   | Default       | Notes                                                                                       |
| ------------------------------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------- |
| `smiles`                                    | str    | *required*    | Validated server-side. Max 512 characters — RDKit's parser overflows the C stack on longer input, and it runs inside the controller process. |
| `project`                                   | str    | `"default"`   | A **bare** name, qualified with your namespace on the way in (`screening` → `alice/screening`) and created on first use. Groups molecules and scopes exports / archives. 1–64 letters, digits, `.`, `_` or `-`; anything else, `/` or `:` included, answers 400. |
| `author`                                    | str    | `"web"`       | **Accepted and ignored.** The provenance label stored as `project_author` in `request_metadata` is always the calling account's username, admin included. The field is still accepted so pre-accounts scripts do not get a 422; the real value comes back in the response. |
| `priority`                                  | int    | `10`          | Higher = served first, and sets the queue allowance: `priority * queue_slots_per_priority` (default 10) waiting SLURM jobs. Ties broken by submission order. Admin only, 1 to 1000 (400 outside); any other account must send the default or nothing, else 400 "Only admin can set priority." Applies to submit, submit-batch and extend. |
| `request_t1`                                | bool   | `false`       | Build a T1 state and run the full chain on it.                                              |
| `request_ox`                                | bool   | `false`       | Build a +1 (oxidised) state.                                                                |
| `request_red`                               | bool   | `false`       | Build a −1 (reduced) state.                                                                 |
| `skip_confsearch`                           | bool   | `false`       | Skip GOAT, send the RDKit-generated geometry straight to optimization.                      |
| `seed_from_s0`                              | bool   | `false`       | T1/ox/red skip their conformer search and start from the lowest S0 conformer once S0 is finished.|
| `request_optimization`                      | bool   | `true`        | If false, the pipeline stops after confsearch.                                              |
| `request_singlepoint`                       | bool   | `true`        | If false, no singlepoint is created (and no vertical excitations either).                   |
| `request_singlepoint_vertical_excitations`  | bool   | `true`        | Adds vert-ox / vert-red / spin-flip singlepoints on the optimised geometry of each state.   |
| `max_conformers_S0`                         | int    | `1`           | Conformer cap for the S0 state.                                                             |
| `max_conformers_T1`                         | int    | `1`           | Conformer cap for the T1 state.                                                             |
| `max_conformers_ox`                         | int    | `1`           | Conformer cap for the ox state.                                                             |
| `max_conformers_red`                        | int    | `1`           | Conformer cap for the red state.                                                            |
| `max_conformers`                            | int?   | `null`        | Legacy override — when set, applies to every state, regardless of the per-state fields. Every `max_conformers*` is 0 or more (0 asks for none); a negative one answers 400, on Extend too. |
| `header_confsearch`                         | str?   | `null`        | Raw ORCA header block (multi-line). Used if no header id is set.                            |
| `header_optimization`                       | str?   | `null`        | Raw ORCA header block. Used if no header id is set.                                         |
| `header_singlepoint`                        | str?   | `null`        | Raw ORCA header block. **Must not contain `Opt` or `Freq`.**                                |
| `advanced` | object? | `null` | Advanced settings per stage: `{"confsearch": {...}, "optimization": {...}, "singlepoint": {...}}`. Keys: confsearch `nprocs`, `maxcore`; optimization `grid`, `scf_convergence`, `geom_convergence`, `ri`, `aux_basis`, `solvation`, `solvent`, `nprocs`, `maxcore`; singlepoint the same without `geom_convergence`. Choices: `grid` DEFGRID1/2/3; `scf_convergence` Normal/Strong/Tight/VeryTight/Extreme; `geom_convergence` Loose/Normal/Tight/VeryTight; `ri` RIJCOSX/RI-J/RIJK/NoRI; `aux_basis` def2/J, def2/JK, AutoAux; `solvation` CPCM/SMD; each also `omit` (ORCA default); `solvent` an ORCA name (letters, digits, `,` `-`, no spaces; max 40); `nprocs` 1–64; `maxcore` 250–16000 MB; `nprocs` × `maxcore` ≤ 400 000 MB. Missing keys take the config defaults (`[advanced.*]`, `GET /api/advanced-defaults`); unknown stages or keys, a wrong value or a non-object answer 400. Rendered into each job: a second `!` line (RI, aux basis, grid, solvation; `Opt Freq` on optimisations, `VeryTightOpt Freq` for geometry VeryTight, which ORCA refuses in `%geom`) and `%maxcore`, `%pal`, `%scf`, `%geom` blocks (SCF Normal as `Convergence Medium`, ORCA's block name for the NormalSCF level). The resolved dict is stored on every state. With `advanced` (or without it, see below) the headers must be method only: a header with `DEFGRIDn`, `*SCF`, RI or aux-basis keywords, `CPCM`/`SMD`, `PALn`, any `Opt`, `Freq`, `KeepDens`, `%pal`, `%maxcore`, `%geom`, `%cpcm` or `%scf Convergence` answers 400 naming the setting (the confsearch header only for `PALn`, `%pal`, `%maxcore`). Without `advanced`, a submission to a project whose S0 states ran on exactly these headers without Advanced settings, the optimisation header itself setting what Advanced owns, stays legacy (headers used as they are, nothing stored); every other one gets the defaults. Resubmitting an existing molecule at the headers of its S0 states with other Advanced method settings (all but `nprocs`/`maxcore`), or in the other mode, answers 400 naming the setting, with or without categories; at other headers it gets new states. CLI: `--opt-grid`, `--sp-grid`, `--opt-scf-convergence`, `--sp-scf-convergence`, `--opt-geom-convergence`, `--opt-ri`, `--sp-ri`, `--opt-aux-basis`, `--sp-aux-basis`, `--opt-solvation`, `--sp-solvation`, `--opt-solvent`, `--sp-solvent`, `--cs-nprocs`, `--opt-nprocs`, `--sp-nprocs`, `--cs-maxcore`, `--opt-maxcore`, `--sp-maxcore`. |
| `header_confsearch_id`                      | int?   | `null`        | ID of a stored `ComputationHeader`. Wins over the raw text version.                         |
| `header_optimization_id`                    | int?   | `null`        | Same.                                                                                       |
| `header_singlepoint_id`                     | int?   | `null`        | Same.                                                                                       |
| `request_spec_uvvis`                        | bool   | `false`       | UV/Vis: a TDDFT singlepoint (`%tddft nroots <uvvis_nroots> tda <uvvis_tda>`, appended to the singlepoint header) on every optimised S0 conformer. The singlepoint header must not already contain `%tddft`, `%cis`, `%eprnmr`, `%esd`, the `NMR` or `ESD` keyword, a frequency keyword (`Freq`/`NumFreq`/`AnFreq`) or an optimisation keyword (`Opt`/`OptTS`/`OptH`/…) — the same rule applies to NMR and ESD. |
| `uvvis_nroots`                              | int    | `20`          | UV/Vis excited states (1–100). Stored only when UV/Vis is requested.                        |
| `uvvis_tda`                                 | bool   | `false`       | Tamm–Dancoff approximation for the UV/Vis TDDFT. Stored only when UV/Vis is requested.      |
| `request_spec_ir`                           | bool   | `false`       | IR: read from the S0 optimisation's frequency calculation — no extra job. A legacy submission needs `Freq` in the optimisation header. |
| `request_spec_nmr`                          | bool   | `false`       | NMR: a singlepoint on every optimised S0 conformer in the `nmr_method`. Shifts are referenced to chloroform or TMS (`nmr_reference`, ¹H/¹³C) and CFCl₃ (¹⁹F), which the pipeline computes itself — once per optimisation/singlepoint header pair, `nmr_method` and Advanced method settings — in the protected `admin/reference_systems` project. Closed-shell molecules only. |
| `nmr_nuclei`                                | list   | `["H","C","F"]` | Nuclei to report (H, C, F, As); stored only with NMR. As: no reference, shielding and EFG. |
| `nmr_method`                                | string | `"singlepoint"` | `"singlepoint"`: the singlepoint header with `NMR` added to its `!` line. `"dh"`: `!revDSD-PBEP86-D4/2021 PCSSEG-3 AUTOAUX NMR NOFROZENCORE TightSCF` plus the singlepoint header's solvation keywords and `%cpcm` only, on its own `nmr_dh_nprocs` × `nmr_dh_maxcore`; refused for elements past Kr (no `pcSseg-3`). Stored only with NMR. |
| `nmr_reference`                             | string | `"chcl3"`       | ¹H/¹³C reference compound: `"chcl3"` (chloroform; its experimental shift in the calculation's solvent is added, Fulmer et al. 2010, e.g. 7.58 / 79.17 ppm in MeCN) or `"tms"`. ¹⁹F always CFCl₃. Stored only with NMR; molecules from before the option read as `"chcl3"`; a different value is no conflict with an existing molecule. |
| `nmr_dh_nprocs`                             | int    | `12`            | Cores of a `dh` NMR job (1–64). Stored only with `nmr_method` `"dh"`; a different value is no conflict with an existing molecule. |
| `nmr_dh_maxcore`                            | int    | `6000`          | MB per core of a `dh` NMR job (1000–16000; cores × (MB + 50) at most 400 GB). The MP2 response needs about 5.8 GB per core already at 30 atoms, whatever the core count; a retry raises it by what ORCA reports missing. Stored only with `nmr_method` `"dh"`. |
| `request_esd` | bool | `false` | Excited-state dynamics: S1 and T1 optimised from the lowest S0 conformer, SOC TDDFT, ORCA ESD rates (ISC, RISC, fluorescence, T1→S0 ISC, phosphorescence; IC with `esd_ic`). Needs a closed-shell singlet, `Freq` in the optimisation header (legacy submissions only), the energy singlepoint, and a functional whose TDDFT gradients ORCA supports (native B88 functionals such as B3LYP need `LibXC(...)`). Molecules with an element past Kr (ECP in def2, e.g. I) need `ZORA`/`DKH`/`X2C` in the singlepoint header. |
| `request_esd_ht` | bool | `false` | Herzberg–Teller where it matters: fluorescence, phosphorescence, and ISC channels whose SOCME is below 5 cm⁻¹ (RISC more than 10 kT uphill stays FC); only with `request_esd`. |
| `esd_ht_scope` | string | `"all"` | With `request_esd_ht`: `"all"` rates, or `"s1"` — fluorescence and S1→Tn ISC only (phosphorescence, T1→S0 ISC and RISC stay FC). Stored only with ESD. |
| `esd_ht_method` | string | `"sp"` | With `request_esd_ht`: HT jobs at the singlepoint header's level (`"sp"`) or the optimisation header's (`"opt"`, e.g. def2-SVP; several times cheaper). Energies and FC couplings stay at the singlepoint level. Stored only with ESD. |
| `imag_freq_tolerance` | float | `0.0` | cm⁻¹, −300 to 0 (else 400). Every optimisation, ESD's S1 included, fails on any imaginary mode below it and is retried as usual; exports, analysis and ESD seeding leave out conformers below it. `0` accepts no imaginary mode; `-10` accepts noise modes down to −10 cm⁻¹. A body still sending `strict_convergence` answers 400. Stored on every state; Extend ignores it and gives the states it adds S0's value. States (and queued submissions) from before the option keep −50 cm⁻¹ (ESD's S1 −100 cm⁻¹ if submitted without the former strict convergence), and so do the states Extend adds to them. CLI `--imag-freq-tolerance`; dashboard: in the "Advanced" section below the headers. |
| `esd_tn_window_ev` | float | `0.3` | S1→Tn ISC is summed over triplets up to this many eV above S1 (0–1). Stored only with ESD. |
| `esd_temperature_k` | float | `298.15` | Temperature of the rates (0 < T ≤ 1000). Stored only with ESD. |
| `esd_ic` | bool | `false` | Also run the S1→S0 internal-conversion rate (harmonic, often unreliable for large gaps). Stored only with ESD. |
| `request_densities` | bool | `false` | Gaussian cube files (electron and/or spin density) from a `%plots` block added to every state's own energy singlepoint — S0, T1, ox, red, not only S0's (ESD's S1 has no energy singlepoint, so it never gets one). Needs the energy singlepoint stage; the singlepoint header must not already contain `%plots`. |
| `density_eldens` | bool | `true` | Write the electron density cube. Stored only with Densities; at least one of `density_eldens` / `density_spindens` must be true. |
| `density_spindens` | bool | `true` | Write the spin density cube — silently skipped on a closed-shell state (multiplicity 1), since ORCA's `orca_plot` cannot produce one there. Stored only with Densities. |
| `density_grid` | int | `75` | Cube grid points per axis (10–300). Stored only with Densities. |
| `density_eldens_file` | str | `"ElDens.cube"` | Cube file name: `NAME.cube`, letters/digits/`_.-` only, max 64 characters; must differ from `density_spindens_file`. Stored only with Densities. |
| `density_spindens_file` | str | `"SpinDens.cube"` | Same rule as `density_eldens_file`, for the spin density cube. Stored only with Densities. |
| `request_singlepoint_nbo` | bool | `false` | Natural population analysis via ORCA's NBO 7: the `NBO` keyword plus a `%nbo` block on every state's own energy singlepoint (same states as Densities, above). Needs the energy singlepoint stage, a singlepoint header that does not already contain the `NBO` keyword or a `%nbo` block, and `[orca].nbo_exe` configured on the server — refused with `400` otherwise (see [`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#densities-and-nbo)). |
| `nbo_keywords` | str | `""` | Extra keywords placed inside `NBOKEYLIST = "$NBO ... $END"`, e.g. `"BNDIDX"`; `ARCHIVE` is added, as ORCA's default list has it; the `FILE.47` it writes stays in scratch. Letters, digits, spaces and `=_.,+-` only (max 200 characters) — no `$`, quotes or newlines. Stored only with NBO. |
| `request_sp_reactivity` | bool | `false` | Reactivity descriptors: the `Hirshfeld` keyword, a `%elprop` block (dipole, quadrupole, polarizability), NBO with `BNDIDX` (Wiberg indices; plus `nbo_keywords` when NBO is also requested) and a kept `.gbw`, on every relaxed and vertical singlepoint; stability analysis on open-shell verticals. About 350–450 GB of `.gbw` per 1,000 molecules. Needs the energy singlepoint stage, a singlepoint header without `Hirshfeld`, `NBO`, `%elprop`, `%nbo` or `STABPerform`, and `[orca].nbo_exe` on the server — refused with `400` otherwise. Submission only; Extend refuses it. CLI: `--reactivity`. |

`request_S1` is **not** exposed: the S1 state exists only as part of ESD (`request_esd`).

**Responses:**

* `200 OK` →
  ```json
  { "id": 42, "smiles": "CCO", "status": "queued",
    "time_created": "2026-06-01T07:14:42.669100" }
  ```
* `400 Bad Request` (invalid SMILES) →
  ```json
  { "detail": "RDKit could not parse 'xxx'.",
    "validation": { "valid": false, "error": "...", ... } }
  ```

**Minimal example — defaults everywhere:**

```bash
curl -X POST http://localhost:8085/api/submit \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"smiles":"CCO","project":"alcohols"}'
# queued as <your username>/alcohols
```

**Full-coverage example — T1/ox/red, vert-ex off, custom per-state conformer counts, header by id:**

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
           "request_singlepoint_vertical_excitations": false,

           "max_conformers_S0": 5,
           "max_conformers_T1": 3,
           "max_conformers_ox": 2,
           "max_conformers_red": 2,

           "header_confsearch_id":    2,
           "header_optimization_id":  3,
           "header_singlepoint_id":   4
         }'
```

Submission never blocks on the cluster: accepting a molecule is one INSERT
into `calculation_entrypoints`. Whether it can start now is the
controller's problem — see *Backpressure and priority* in the README.

### `POST /api/submit-batch`

Queue many molecules under one set of options. Takes every field
`POST /api/submit` takes, except that `smiles` is replaced by
`smiles_list` (max 10 000 entries). Use this for anything library-sized: it
is a single transaction, where N single submissions are N commits and N
rounds of lock contention with the controller. Every SMILES is validated
first, and the database write lock is taken only for the final insert.

Invalid SMILES do **not** fail the request. Each one is reported
individually — including any entry over the 512-character bound — so the
caller can log what was refused and keep the rest:

```json
{
  "queued":   [{"id": 41, "smiles": "c1ccc2[nH]ccc2c1"},
               {"id": 42, "smiles": "CCO"}],
  "rejected": [{"smiles": "not-a-smiles",
                "detail": "RDKit could not parse 'not-a-smiles'."},
               {"smiles": "C[C]1CC(C#N)C1",
                "detail": "T1 requires a closed-shell reference, but ..."}],
  "counts":   {"queued": 2, "rejected": 2}
}
```

`400` is returned for the whole batch only when `smiles_list` is empty, or
a field is refused as for `POST /api/submit` (`advanced`, a header,
`project`, `priority`, `imag_freq_tolerance`, a negative `max_conformers*`).

### `GET /api/advanced-defaults`

The config defaults and choices of the Advanced settings, for the submit
form: `{"advanced": {confsearch, optimization, singlepoint}, "choices":
{grid, scf_convergence, geom_convergence, ri, aux_basis, solvation},
"imag_freq_tolerance": 0.0}`. Any signed-in account.

```bash
curl -X POST http://reaction.uni-muenster.de:60001/api/submit-batch \
     -H "Content-Type: application/json" \
     -H "X-AutoDFT-API-Key: $AUTODFT_API_KEY" \
     -d '{"smiles_list": ["c1ccc2[nH]ccc2c1", "CCO"],
          "project": "heteroarenes", "priority": 1,
          "request_t1": true, "request_ox": true, "request_red": true,
          "skip_confsearch": true}'
```

---

## 2. Status & monitoring

Every endpoint in this section counts and lists **only the caller's own
work** — molecules, tasks, jobs and queued entrypoints in projects they
own. Admin sees everything. For cluster-wide health without anyone's
data, see `GET /api/cluster` in §0.

### `GET /api/overview`

```json
{
  "molecules": 4,
  "tasks":   {"created": 0, "pending": 16, "successful": 0, "failed": 0},
  "jobs":    {"RUNNING": 14, "PENDING": 2, "COMPLETED": 0},
  "job_groups": {"running": 14, "pending": 2},
  "queue_length": 0
}
```

`job_groups` counts the groups `/api/jobs?status=` takes.

### `GET /api/queue`

Entrypoints that haven't been processed yet (`time_started IS NULL`).

### `GET /api/entrypoints/failed`

Entrypoints that raised a `processing_error` (e.g. SMILES that slipped
past validation, RDKit/OpenBabel both unavailable). Surface these to the
user — the controller never silently retries them.

```json
[
  { "id": 7, "smiles": "weird ?? input",
    "priority": 10,
    "time_created":  "...",
    "time_started":  "...",
    "processing_error": "RuntimeError: Cannot generate 3-D geometry for ..." }
]
```

### `GET /api/molecules` `?project=&limit=&offset=`

Paginated molecule list.

### `GET /api/molecules/{id}`

Full molecule tree: states → tasks → jobs. An optimisation task judged
since the broken-bond check existed adds `connectivity` (below).

### `GET /api/tasks` `?status=&type=&limit=`

`status` is one of `created | pending | successful | failed`; `type` is
one of `confsearch | optimization | singlepoint | singlepoint_vert_ox |
singlepoint_vert_red | singlepoint_vert_spin_change | singlepoint_nbo |
singlepoint_uvvis | singlepoint_nmr | singlepoint_soc | esd_isc | esd_risc |
esd_ic | esd_fluor | esd_isc_t1s0 | esd_phosp`. Any other value answers 400
listing the valid ones.

### `GET /api/jobs` `?status=&limit=`

`status` matches SLURM (`RUNNING | PENDING | COMPLETED | FAILED |
TIMEOUT | CANCELLED | UNKNOWN`), or names a group: `running` (RUNNING,
COMPLETING) or `pending` (PENDING, CONFIGURING, SUBMITTING), the
dashboard's Running and Pending. `limit` defaults to 50, at most 500.
The body is the newest `limit` matching jobs; the `X-Total-Count`
header says how many match in all.

---

## 3. Projects

`{name}` in every route below is a project written `owner:project` —
`/api/projects/admin:phenols`. A bare `{name}` still works and means
"mine"; for admin it resolves to the unique project with that bare name,
or answers **409** listing the candidates when two owners share it.

### `GET /api/projects`

Names come back **qualified**, which is the form the other routes and the
CLI expect. The list is derived from the molecules that exist, so a
project whose entrypoints are all still queued does not appear yet —
`GET /api/whoami` lists it from the moment it is created.

```json
[
  { "name": "admin/phenols", "molecules": 12,
    "tasks_total": 192, "tasks_failed": 3, "tasks_successful": 145,
    "archived": false, "protected": false }
]
```

`archived` is true once every molecule in the project has been through
`/archive` (its raw files are gone; the rows stay). `protected` marks
`admin/default`, the one project that may never be wiped or archived.

### `GET /api/projects/{name}`

Per-project view: status, progress, success rate, and one row per
molecule:

```json
{
  "name": "admin/phenols",
  "status": "running",
  "archived": false,
  "protected": false,
  "in_flight_molecules": 1,
  "in_flight_tasks": 4,
  "completed_molecules": 11,
  "total_molecules": 12,
  "submission_progress": { "total": 12, "started": 12 },
  "success_rate":        { "total_molecules": 12, "successful_molecules": 11 },
  "molecules": [
    { "id": 3, "smiles": "Oc1ccccc1", "states": 4, "tasks": 16,
      "successful": 16, "failed": 0, "in_flight": 0, "done": true,
      "created_at": "2026-06-01T..." }
  ]
}
```

`status` is one of `empty | running | complete | complete_with_failures`
— "complete" meaning every task has reached a terminal state, not that
every one succeeded.

### `GET /api/projects/{name}/molecules-detail`

The same molecules, one level deeper: each state (S0 / T1 / ox / red)
with its confsearch status and one row per conformer carrying the status
of that conformer's optimization and of every singlepoint hanging off it,
plus `esd`: one combined status (`failed` > `pending` > `created` >
`successful`, `null` when there are none) over that conformer's
`singlepoint_soc` and every `esd_*` rate task. This is what the
dashboard's *Project Overview → Molecules* subpage renders.

`index` is the conformer number: the 1-based position among the state's
successful optimisations by id, as in the exports. An optimisation that
has not succeeded is listed with `index: null`.

A conformer whose optimisation was judged since the broken-bond check
existed adds `connectivity`: `{"ok": false, "lost": true, "fragments": 2,
"detail": "dissociated: 2 fragments (C4H8F + H); C-H bonds 9 -> 8"}`
(`ok: true`, `detail: ""` when intact; `lost: false` with `bond formed: ...`
when only an extra bond appeared). Older conformers carry no key. See
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#broken-bonds).

### `GET /api/projects/{name}/sp-energies`

`{"name": ..., "tasks": {"<task_id>": {"energy": -1234.56, "gibbs": -1234.33}}}`
for every successful singlepoint except NMR and UV/Vis. `energy` is the
final single-point energy in Eh. `gibbs` is set on a relaxed `singlepoint`
only: `energy` + the raw G−E(el) of the successful optimisation it depends
on (no quasi-RRHO), `null` without one. Singlepoints with no stored or
parsable energy are left out.

### `GET /api/projects/{name}/sp-densities`

`{"name": ..., "tasks": {"<task_id>": {"eldens": true, "spindens": false, "build": {"eldens": true, "spindens": false}}}}`
for the successful singlepoints whose molecule or state requested densities
and whose latest successful job folder holds the cube, or the
`.gbw` + `.densities` + `.densitiesinfo` orca_plot rebuilds it from. SpinDens
only for an open-shell job. `build` says which kinds orca_plot must make first.

### `POST /api/projects/{name}/densities/{task_id}/{kind}`

`kind` is `eldens` or `spindens`. Returns the cube (`text/plain`,
`<task_id>_<kind>.cube`), building it with orca_plot first and caching it in
the job folder. At most two builds run at once; `503` while none is free.
`404` when densities were not requested, the kind was off at submission, a
closed-shell job is asked for SpinDens, or the job folder is gone. `409`
(`Export and Delete is running for this project.`) while the project's
archive runs. An archive that starts during a build waits for it.

### `GET /api/projects/{name}/tasks/{task_id}/files/{kind}`

`kind` is `out` (the `output.out` of a successful optimisation or
singlepoint) or `xyz` (the final geometry of a successful optimisation),
from the task's latest successful job. Named
`mol<id>_<state>_conf<k>_<stem>.<kind>`, as in the export tree (so
`<state>__state<state_id>` for one of two states of one name). `400` for
another kind, `404` for a task that is not a successful opt/SP of the
project, for `xyz` on a singlepoint, and when the file is no longer on disk.
`409`, as for the cube, while the project's archive runs; an archive that
starts during a download waits until the file is sent or the client leaves.

### `GET /api/projects/{name}/state-analysis`

Triplet energies, redox free energies / E vs SCE in MeCN, and 4-point
Marcus reorganisation energies, for both the `lowest_energy` and the
`rmsd_matched` conformer-selection modes. E vs SCE needs the optimisation
and the singlepoint of both S0 and ox/red in MeCN, read from each state's
headers as rendered with its Advanced settings. Otherwise ox/red report ΔG
only, with a `note` naming each state and stage not in MeCN, whenever one
of the four ran in MeCN or the project's `solvation_mecn` is true (some
state's optimisation and singlepoint both ran in MeCN). A pick
whose optimisation has a broken-bond check adds `connectivity` (same shape
as in `molecules-detail`). One entry per method family, a molecule's states
at one header set: a molecule resubmitted at other headers has an entry for
each, and S0, T1, ox and red are paired only within one. Each entry carries
`state_id` (the family's S0 state, `null` without one) and `header_ids`
(`[confsearch, optimization, singlepoint]`); the dashboard names them on the
card when a molecule has more than one.

The payload is cached per project, and rebuilt when a task, a stored job
result (also one `backfill-results` rewrites), molecule or state metadata,
or the text of a header the project's states use changes.

### `POST /api/projects/{name}/state-analysis/export`

Builds the same payload as a multi-sheet XLSX in a background job: returns
`202` with the job (`409` while one is already running). Poll
`GET /api/projects/{name}/jobs`, then fetch the file from
`GET /api/jobs/{id}/download`. Sheets: Summary, Lowest
Energy, RMSD Matched, Conformers). Energies in Hartree, potentials in V
vs SCE. When any pick has a broken-bond check, the mode sheets add a last
column `broken_bonds` (`<state>: ok`, `<state>: <detail>` or
`<state>: unchecked` per pick, joined by ` | `) and
Conformers a last column `connectivity`. When any ox/red has a `note`, the
mode sheets add `ox_sce_note` and `red_sce_note` after those. When a
molecule has more than one method family, the mode sheets and Conformers
add `S0_state_id`, `confsearch_header_id`,
`optimization_header_id` and `singlepoint_header_id` after those, and the
Summary's Molecules counts each molecule once.

### `GET /api/projects/{name}/photophysics` `?molecule_id=`

UV/Vis, IR, NMR, ESD and NBO for every molecule submitted with those
categories (Densities has no payload of its own — see
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#densities-and-nbo) for its cube
files). A plain resubmission of an existing molecule with a category it
does not have, or with different options for one it already has (or other
Advanced method settings at the same headers), answers 400; add categories to existing molecules with
[`POST /api/projects/{name}/extend`](#post-apiprojectsnameextend).

Without `molecule_id`, one summary per molecule (this view is cached, and
refreshed when this project or the NMR reference project changes — true
for live molecules only; an archived molecule is served from its frozen
payload and does not pick up a reference that finishes afterwards, see
below):

    {"project": "nho/p", "temperature_k": 298.15, "molecules": [
      {"id": 7, "smiles": "c1ccccc1", "state_id": 21, "archived": false,
       "uvvis": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
                 "unweighted": 0, "weighting": "G", "shortest_nm": 156.5,
                 "peak": {"wavelength_nm": 229.6, "energy_ev": 5.4, "fosc": 0.24}},
       "ir": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
              "unweighted": 0, "weighting": "G", "scaled": false, "scale_factor": 1.0,
              "peak": {"frequency_cm": 410.2, "intensity_km_mol": 12.3}}}]}

`count` is conformers with a spectrum and the energy the weights use;
`pending`, `failed`, `unavailable`, `unweighted` account for the rest
(job still running or its energy singlepoint still running, job failed,
output missing/unparsable, or the spectrum is in but not the energy the
weights use and none is coming). `weighting` names the energy scale
behind the Boltzmann weights: `"G"` when a conformer has a thermal
correction, else `"E_sp"`, else `"equal"`. `peak` is the transition/mode
with the largest weight × fosc (or × intensity), or `null` when `count`
is 0; it is present for UV/Vis and IR only. UV/Vis also reports
`shortest_nm`, the shortest wavelength across every counted transition.
IR frequencies are unscaled harmonic ones (`"scaled": false`,
`"scale_factor": 1.0`); the dashboard's scale factor (default 0.95)
changes only its plot.

An NMR molecule's entry adds `nmr`, with the same counts and no `peak`:

    "nmr": {"count": 1, "pending": 0, "failed": 0, "unavailable": 0,
            "unweighted": 0, "weighting": "G", "equivalence": "topological",
            "signals": {"H": 1, "C": 1},
            "reference": {"H": {"compound": "C[Si](C)(C)C", "status": "ok",
                                "molecule_id": 12, "sigma_ppm": 31.354,
                                "method_matches": true}, "C": {...}}}

`signals` is the number of distinct signals per requested nucleus
(`nmr_nuclei`) present in the molecule. As has no reference: its signals carry
`shift_ppm: null`, the shielding and its anisotropy, and the Boltzmann-weighted
electric field gradient (`efg_cq_mhz`, `efg_eta`, `efg_vzz_au`). `equivalence` is `"topological"`
when atoms are grouped by the symmetry classes of the bonds perceived from
the first counted conformer's geometry, else `"none"` (one signal per
atom). `reference` names, per reported nucleus, the reference compound
(chloroform or TMS for H and C per `nmr_reference`, CFCl₃ for F) at the molecule's optimisation and
singlepoint headers and NMR method in `admin/reference_systems` — the one
whose NMR input matches the molecule's, else the one stored with the same
`nmr_method` (none for molecules from before the option) and Advanced method
settings: `status` is `"ok"`,
`"pending"`, `"failed"` or `"missing"` (none at this method); `sigma_ppm`
is its mean isotropic shielding for that element; `delta_ref_ppm` the
reference's experimental shift added to σ_ref − σ (0 for TMS and CFCl₃) and
`delta_ref_solvent` the NMR solvent it was taken in (`null` for TMS/CFCl₃);
`method_matches` says
whether its NMR input's `!` keywords equal the molecule's (SCF convergence
and `PALn` ignored), `null` if either input is missing. `signals` and
`reference` stay empty until a conformer is counted. `reference_systems`
is a reserved project name, refused for every submitter.

A molecule entry with no conformer left to show (every optimisation
failed, or none has run yet) adds `stage`: `"searching"` while work is
still open for that state, else `"none"`. ESD needs no conformer pool
(it works from S1/T1, not S0 conformers), so an ESD-only molecule never
gets `stage`.

An ESD molecule's entry adds `esd`:

    "esd": {"status": "done", "temperature_k": 298.15, "herzberg_teller": false,
            "tn_window_ev": 0.3, "seed_task_id": 118,
            "rates": {"isc": {"status": "successful", "rate_s": 9521.34, "ht": false,
                              "ht_reason": "Herzberg-Teller not requested",
                              "diagnostics": [{"triplet": 1, "k_inlinew_100": 9521.34,
                                               "k_inlinew_25": 8810.2, "k_inlinew_250": 10240.7,
                                               "k_tcutfreq_300": 9120.5,
                                               "width_sensitive": false,
                                               "soft_mode_dominated": false}]},
                      "risc": {"status": "successful", "rate_s": 1.236e-05},
                      "ic": {"status": "successful", "rate_s": 26472.1},
                      "fluorescence": {"status": "successful", "rate_s": 13386.15, "e00_ev": 2.3955},
                      "isc_t1_s0": {"status": "successful", "rate_s": 0.1599},
                      "phosphorescence": {"status": "successful", "rate_s": 42.84}},
            "delta_est_ev": 0.6657, "delta_est_uks_ev": 0.5078,
            "derived": {"tau_s1_ns": 20251.3, "phi_fluorescence": 0.2711,
                        "phi_isc": 0.1928, "phi_ic": 0.5361,
                        "tau_t1_us": 23255.6, "phi_phosphorescence": 0.9963,
                        "phi_isc_t1_s0": 0.0037, "phi_risc": 0.0000002874},
            "flags": ["isc T2: a negative rate (-1.54e-09 s⁻¹) was set to 0."]}

`status` is `"waiting"` before the S1/T1 states are seeded (or while
seeding waits for every S0 conformer to finish), `"running"` while any
rate is still open, `"done"` once all six have settled (or are not requested), or `"failed"`
(with `reason`) if seeding itself failed. Each entry of `rates` is one of
`isc`, `risc`, `ic`, `fluorescence`, `isc_t1_s0`, `phosphorescence`, with
its own `status`: `"not requested"` (`ic` without `esd_ic`), `"successful"`, `"created"`, `"pending"`, `"failed"`
(with `reason`), `"waiting"`, `"blocked"` (with `reason` — an upstream
optimisation or SOC singlepoint failure, named with the `!LibXC(<functional>)`
fix when that is the cause, or an S1/T1 optimisation whose geometry lost
bonds: `T1 geometry dissociated: ... (optimisation task N); no rates on a
broken geometry`), `"unavailable"` (the output has no rate
to parse), or `"incomplete"` (`rate_s` null, `reason` naming the ORCA
jobs that printed no rate: a multi-job rate is never combined from a
subset). A successful rate's `rate_s` sums its jobs for `isc` (one job
per T_n channel within `tn_window_ev` of S1) and averages them for every
other rate; `fluorescence` also reports `e00_ev`, its 0-0 energy. An
FC-mode `isc`, `isc_t1_s0` or `risc` rate whose SOCME printed as
`0.00` cm⁻¹ (below ORCA's print precision) adds `socme_zero: true` —
its rate is 0 because the channel is symmetry- or El-Sayed-forbidden,
not because it is truly closed; `request_esd_ht` is the remedy.
`herzberg_teller` reports whether HT was requested; each built rate's
`ht` and `ht_reason` say whether it actually ran HT and why (per channel
for `isc`, e.g. `"T1: SOCME 0.88 cm-1 < 5; T2: SOCME 20.00 cm-1 >= 5"`),
and the detail view's `jobs` rows carry the same per job. `isc`,
`isc_t1_s0` and `risc` add `diagnostics`, one entry per channel with the
FC reruns' rates (`k_inlinew_100`, `k_inlinew_25`, `k_inlinew_250`,
`k_tcutfreq_300`; null for a rerun that printed no rate) and the verdicts
`width_sensitive` (25 vs 250 cm⁻¹ more than 3× apart) and
`soft_mode_dominated` (TCUTFREQ 300 below a tenth of the baseline);
channels whose rates stay below 1 s⁻¹ are never flagged. The verdicts
describe the FC lineshape only — an HT channel's rerun uses its total
SOCME, often ~0, and is then not judged. These reruns never enter
`rate_s`, and a failed rerun never fails the rate. `delta_est_ev` is the TDDFT ΔE(S1-T1);
`delta_est_uks_ev` instead compares the TDDFT S1 against the T1 state's
own (UKS) energy singlepoint at the T1 geometry. `derived` reports
`tau_s1_ns` / `phi_fluorescence` / `phi_isc` / `phi_ic` once
`fluorescence`, `isc` and `ic` are all in (with `ic` not requested:
without `phi_ic`, from `fluorescence` and `isc`, flagged), and `tau_t1_us` /
`phi_phosphorescence` / `phi_isc_t1_s0` / `phi_risc` once
`phosphorescence`, `isc_t1_s0` and `risc` are all in. `flags` lists
warnings: a negative rate clamped to 0, a job whose sum of K*K exceeds 7
(geometries too far apart for a reliable harmonic rate), a zero-SOCME FC
channel (see `socme_zero` above), and, when that channel feeds a
`derived` group, that its lifetime/yields count the rate as 0 too, and a
width-sensitive or soft-mode-dominated channel.

An NBO molecule's entry adds `nbo`, one entry per NBO-flagged state that
has its own energy singlepoint — S0, and T1 / ox / red when requested;
ESD's S1 never appears, since it has none. This covers only the states
submitted together with the entry's own S0 (same `confsearch_header_id`,
`optimization_header_id` and `singlepoint_header_id`); a molecule
resubmitted with different headers gets a separate entry with its own
states, never mixed with the first:

    "nbo": {"states": [
      {"count": 1, "pending": 0, "failed": 0, "unavailable": 0, "unweighted": 0,
       "weighting": "G", "state": "S0", "state_id": 21,
       "extremes": {"most_negative": {"index": 3, "element": "O", "charge": -0.612},
                    "most_positive": {"index": 0, "element": "C", "charge": 0.812}}}]}

Each state's counts and `weighting` mean the same as UV/Vis/IR/NMR, but
weighted on that state's own conformers rather than only S0's. `extremes`
is the Boltzmann-weighted most negative / most positive natural charge,
`null` until at least one conformer's charges are in. `state_id` is that
state's own id — for T1/ox/red this differs from the entry's `state_id`
above, which is always the S0 state's. States are ordered S0, T1, ox,
red. An NBO-only molecule (no UV/Vis, IR, NMR or ESD) never gets `stage`;
its states' own `pending`/`unweighted` counts show whether conformers are
still in.

With `?molecule_id=`, each successful rate also adds `jobs` (its
per-ORCA-job values — `rate_s`, `fc_percent`, `ht_percent`, `k_squared`,
`e00_cm`, plus whatever went into building it; an HT job's `socme_cm` is
its own sublevel's |SOC| as ORCA printed it), and `esd` adds
`energies_eh` (E(S0), E(S1), E(T1)), `socme_cm` (`S1_T1_at_T1`,
`S1_T1_at_S1`, `T1_S0_at_S0`) and a flag for any state whose optimisation
still shows a soft imaginary mode.

With `?molecule_id=`, that molecule's entries add `conformers` (read on
request, not cached) and drop nothing:

    "uvvis": {..., "conformers": [{"conformer_index": 1, "opt_task_id": 90,
              "degeneracy": 3, "weight": 1.0, "transitions": [{"root": 1, "energy_ev": 5.4,
              "wavelength_nm": 229.6, "fosc": 0.24}]}]}

    "nmr": {..., "conformers": [{"conformer_index": 1, "opt_task_id": 90,
            "degeneracy": 3, "weight": 1.0}],
            "nuclei": {"H": [{"atoms": [4, 5], "count": 2,
                              "shielding_ppm": 22.614, "shift_ppm": 8.74}],
                       "C": [...]}}

Each NMR signal is one symmetry class: `atoms` are 0-based ORCA atom
indices (the dashboard and the XLSX show them 1-based, like NBO), `shielding_ppm` is the Boltzmann-weighted isotropic shielding
averaged over them, and `shift_ppm` = `sigma_ppm` − `shielding_ppm`, or
`null` unless the reference's `status` is `"ok"`. Signals are sorted by
ascending shielding (descending shift).

Each of `nbo`'s states adds `atoms` (the weighted natural charge, and
`spin` for an open-shell state, per atom) and its own `conformers`:

    "nbo": {"states": [{..., "atoms": [{"index": 0, "element": "C", "charge": 0.812},
                        {"index": 1, "element": "O", "charge": -0.612, "spin": 0.034}],
                        "conformers": [{"conformer_index": 1, "opt_task_id": 90, "degeneracy": 3, "weight": 1.0,
                                        "charges": [{"index": 0, "element": "C", "charge": 0.812,
                                                     "spin": null}, {"index": 1, ...}]}]}]}

`index` is the 0-based ORCA atom index (NBO's own atom numbering minus
one). `spin` (Natural Spin Density) is present only on an open-shell
state's atoms; a closed-shell state's atoms carry no `spin` key.

`conformer_index` is the conformer's 1-based position among the state's
successful optimisation tasks by id, the same numbering as the CSV, the
export tree and the Molecules page.
`degeneracy` is GOAT's g of the conformer the optimisation started from (1
without one); `weight` ∝ g·exp(−E/kT). A conformer whose optimisation was
flagged by the broken-bond check adds `connectivity_ok: false`; see
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md#weighting).

A molecule any of whose state family's optimisations lost bonds adds
`broken_bonds` (summary and detail): `[{"state": "T1", "state_id": 23,
"conformer_index": 1, "opt_task_id": 95, "detail": "dissociated: ..."}]`.

### `GET /api/projects/{name}/methods`

What the project ran with, for checking. States are grouped by their
three header ids, their Advanced settings, the molecule's S0 request
flags and category options, and the imaginary-frequency tolerance:

    {"groups": [{
       "molecules": 949, "states": {"S0": 949, "T1": 949},
       "headers": {"confsearch": "!GOAT XTB ...", "optimization": "!wB97X-D3 def2-SVP\n",
                   "singlepoint": "!wB97X-D3 def2-TZVP\n"},
       "advanced": {"confsearch": {...}, "optimization": {...}, "singlepoint": {...}},
       "settings": {"optimization": {"grid": {"value": "DEFGRID3", "source": "setting"},
                                     "geom_convergence": {"value": "Normal", "source": "ORCA default"}, ...}},
       "imag_cutoff": {"S0": {"value": -50.0, "rule": "legacy: -50"}},
       "categories": {"request_sp_reactivity": true, "max_conformers": {"S0": 5}, ...},
       "inputs": [{"task_type": "singlepoint_vert_ox", "multiplicity": 2, "charge": 1, "states": ["S0"],
                   "input": "!wB97X-D3 def2-TZVP\n...", "retried_jobs": 3}]}],
     "pipeline": {"max_attempts": 3, "retry": {"IncreaseResources": "..."},
                  "stages": {"optimization": {"time_limit": "1-00:00:00", "default_nprocs": 16,
                                              "default_mem_per_core": 1000}, ...},
                  "slurm": {"partition": "CPU", "nice": 1000},
                  "orca": {"path": "orca", "mpi_env": ""},
                  "orca_versions": {"optimization": "6.1.0"}}}

* `settings` has one row per Advanced key of each stage. `source` is
  `setting` (the state's Advanced settings), `header` (a legacy state,
  read from the header text) or `ORCA default` (value `DEFGRID2` for the
  grid, `Normal` for the geometry convergence, `null` otherwise). ESD
  optimisations (S0 with ESD, S1, T1) run at least Tight: the geometry
  convergence then shows that level with source `ESD floor`, or
  `ESD floor (S0, S1, T1)` naming them when the group holds other states.
  `advanced` is `null` for a legacy group.
* `imag_cutoff` is per state description; `rule` is `imag_freq_tolerance`,
  `legacy: -50`, or `legacy: -100 (ESD S1 with strict_convergence off)`.
* `inputs` holds the full `input.inp` a fresh job gets, per task type, job
  multiplicity (closed- vs open-shell verticals), job charge (ox vs red)
  and ESD role present in the group, rendered by the job-generation code
  without writing files. A retry may modify it; `retried_jobs` counts the
  group's jobs of that type that ran as retries. An input that cannot be
  built shows `Cannot render: <reason>`.
* `pipeline` is the controller's current config, not what earlier jobs
  ran with; `orca_versions` is read from the newest successful output of
  each task type (`null` when that file is gone).

### `GET /api/projects/{name}/extend-info`

What the project's unarchived molecules already have, for Extend:
`molecules`, `archived`, `extendable` (with a `reason` when not: an empty or
fully archived project, or molecules on more than one header set or set of
Advanced method settings),
`headers` (`confsearch` / `optimization` / `singlepoint`: `{id, text}`),
`advanced` (the first S0 state's; `null` for a legacy project),
`categories` and `states` (`T1`, `ox`, `red`) as `"all"` / `"some"` /
`"none"` of the molecules, the `options` of each present category (ESD
also carries `request_esd_ht`), `max_conformers` per existing state,
`settings` (`request_confsearch`,
`request_singlepoint_vertical_excitations`), `submit_only` (categories that
cannot be added later) and `smiles`. Owner or admin.

### `POST /api/projects/{name}/extend`

Adds calculations to the project's molecules — all unarchived ones, or
those in `smiles_list` — without recomputing anything. The body is a
[`submit-batch`](#post-apisubmit-batch) body: tick the categories and states
to have (present ones may stay ticked); headers in the body are ignored and
the project's own are used, and a body with `advanced` answers 400: each
extension takes its S0 state's Advanced settings, and a legacy project stays
legacy. Each molecule gains only what it lacks:
T1/ox/red states (with their conformer search), UV/Vis and NMR singlepoints
for its finished S0 conformers (running ones get them when they finish),
IR, and ESD (S1 plus a triplet seeded from the lowest S0 conformer —
`T1_esd` when the molecule has its own T1). Categories a molecule already
has stay as they are, whatever options the body carries; added ones take the
body's options. ESD added this way keeps the S0 conformers' own `Opt` level
(no TightOpt). A molecule is rejected when it is archived, not in the
project, over 512 characters, or asks for densities or NBO it does not have
(they run inside the energy singlepoints: first submission only); one with
nothing to add is `skipped`, and so is one an unprocessed extension
already covers (same project, categories and states; a repeat in the body
too). ESD needs the S0 Hessians: every optimisation keeps its
`input.hess` since 2026-09-28, and a lowest S0 conformer without one stops
the molecule's ESD before S1 and T1 start, with that reason. Answers
`{project, queued: [{id, smiles}], rejected: [{smiles, detail}], skipped:
[smiles]}`; 400 when the project is not extendable, or, for anyone but
admin, when a header of its S0 states was deleted ("A header of this project
was deleted by admin; …"); 409 while an archive or export job runs. The dashboard's project page opens it as
"Add calculations…": the submission form with the project's molecules and
headers, present items greyed out. Owner or admin.

### `POST /api/projects/{name}/export` `?format=csv|json|files|xlsx|photophysics|reactivity&all_conformers=true|false&keep_gbw=true|false`

Starts an export as a background job and answers **202** for every
format — none of them block the request. Writes into
`<export_data>/<owner>/<project>/`, with the **bare** project name as
the filename stem:

* `csv`   → `<project>.csv` (summary table of energies; when a molecule
  holds two states of one name, every row adds `state_id`,
  `confsearch_header_id`, `optimization_header_id`, `singlepoint_header_id`,
  as does `json`)
* `json`  → `<project>.json`
* `files` → the export tree, the dashboard's "Export Project" (below);
  `all_conformers` writes every valid conformer's folder instead of only
  the reported one per state, `keep_gbw` adds singlepoint `.gbw` files
* `xlsx`  → `<project>_state_analysis.xlsx`
* `photophysics` → `<project>_photophysics.json` (the full UV/Vis, IR,
  NMR, ESD and NBO detail payload) plus `<project>_photophysics.xlsx`
* `reactivity` → `<project>_reactivity.xlsx` plus `<project>_reactivity.json`
  (the same rows), for scripts; Export Project writes them too. One row set
  per successful relaxed and vertical singlepoint of each Reactivity state's
  reported conformer (every valid conformer with `all_conformers`):
  * `molecular`: mol_id, smiles, state, state_id, conformer, opt_task_id,
    `sp_kind` (relaxed / vert_ox / vert_red / vert_spin_change),
    sp_task_id, charge, multiplicity, `energy_Eh`, `polarizability_iso_au`,
    `quad_xx_au` … `quad_yz_au`, `s2`, `stable`, `stability_lowest_Eh`,
    `note` (`unchecked`, or `no reactivity output` for a singlepoint
    without the Reactivity blocks)
  * `atomic`: the identifying columns plus `atom_1based`, `element`,
    Hirshfeld and NPA charge and spin
  * `wiberg`: the identifying columns plus `i_1based`, `j_1based`,
    `wiberg_index` (pairs ≥ 0.05)

```json
{ "project": "admin/phenols",
  "job": { "id": 42, "qualified_name": "admin/phenols", "owner_id": 3,
           "kind": "export_csv", "status": "running",
           "params": {"all_conformers": false},
           "result": null, "error": null,
           "created_at": "2026-09-24T10:00:00+00:00",
           "started_at": "2026-09-24T10:00:00+00:00", "finished_at": null } }
```

Poll `GET /api/projects/{name}/jobs` for the job's `status` and, once
`successful`, its `result` (the written path); download the file via
`GET /api/jobs/{id}/download`. A `csv` or `json` export with no results to
write removes the stale file at its target and answers
`{"format": "csv", "message": "Nothing to export", "downloadable": false}`.

`404` when the project holds no molecules, also when a wipe has emptied it
since the request looked; `409` when a job is already in flight for the
project (detail `Export and Delete is running for this project.` while its
archive runs) or while a wipe of it, or a database reset, is under way.
`csv` / `json` / `files` also answer `409`
for an archived project — their source files are no longer on disk —
and so does `reactivity` (the archive's tree already holds its workbook).
`xlsx` and `photophysics` are allowed on an archived project: `xlsx` is
built from the archive's all-conformers CSV, and `photophysics` is served
from the payload archiving froze — see [`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md).

**The export tree** (`files` and the archive), replacing a previous one in
the same directory (only the folders of the molecules exported now are replaced;
archived or wiped molecules keep theirs, and a molecule whose id is still held by
a wiped one's folder is written to `molecules/<mol_id>_2/`):

```
<export_data>/<owner>/<project>/
  <project>.csv                  # reported conformer per state
  <project>_all_conformers.csv   # every conformer, excluded ones with their note
  <project>_state_analysis.xlsx
  <project>_photophysics.xlsx    # with .json, only for a project with photophysics molecules
  <project>_reactivity.xlsx      # with .json, only for a project with Reactivity molecules
  molecules/<mol_id>/
    molecule.json                # SMILES, states, reported conformer, task ids, job_path per file
    S0/confsearch.out  S0/confsearch_ensemble.xyz
    S0/conf1/opt.inp opt.out opt.xyz  sp.inp sp.out [sp.gbw]  vert_ox.inp vert_ox.out
             uvvis.inp uvvis.out  nmr.*  esd_fluor.*  esd_isc_part1.*  sp_ElDens.cube …
    T1/ ox/ red/ …
    S0__state<state_id>/ …       # instead of S0/, for each of two states of one name
```

Only each successful task's latest successful job is copied, and of it only
the input, output and (optimisations) final geometry; never failed attempts,
SLURM logs or trajectories. `.gbw` files come only with `keep_gbw`, cubes
only for a Densities state. Conformers are numbered as in the CSV;
a conformer with an imaginary mode below its state's cutoff gets no folder
and is named, with its note, in `<project>_all_conformers.csv` and
`molecule.json`.

### `POST /api/projects/{name}/archive`

**Destructive** — the dashboard's "Export and Delete". Writes the export
tree (above), then goes through the project's molecules one at a time:
it flags the molecule `archived = true` and then deletes its
`<comp_data>/mol_<id>/`. An archive interrupted by a restart or an error
leaves the project partly archived, and re-running it is safe. It keeps
the export folders (and frozen photophysics payloads) of the molecules
already archived, deletes any raw folder they have left, and exports and
deletes only the rest. A molecule not flagged whose raw data is gone (none
of its job folders exists; an archive cut short under older code) keeps its
export folder as it is and is flagged; a plain export also keeps it, and
the log names it. The controller also deletes, when it next starts,
any raw folder an archived molecule has left, which covers an archive killed
on its last molecule (the project is then archived and cannot be re-run).
It does so only when the project's export folder exists and one of the
molecule's job paths lies in that folder (so a database paired with another
`comp_data` deletes nothing), and not while the project has a job running;
a folder it keeps is named in the log. The database rows are **kept**, so the
project stays listed and browsable as read-only history — what is gone is
the raw tree on disk, which is why an archived project can no longer be
exported. Everything not in the tree (other conformers when only the
reported one is chosen, `.gbw` files unless kept, failed attempts, logs) is
deleted with it. Answers **202** with the job, like the exports.

**Body:**

```json
{ "confirm": "alice/screening", "all_conformers": false, "keep_gbw": false }
```

`confirm` is required and must equal the qualified project name exactly, as
for the wipe; a missing or different one answers **400** and nothing starts.
The dashboard asks for the name to be typed back. Any other field answers
**422**.

**Job result:**

```json
{ "molecules": 12, "files_copied": 96, "archived": 12,
  "root": "/.../export_data/admin/phenols",
  "downloadable": false, "photophysics_frozen": 5 }
```

`molecules` and `files_copied` count what this run exported; `archived`
counts all of the project's archived molecules, including those a cut-short
earlier run already archived.

`photophysics_frozen` counts the molecules whose UV/Vis, IR, NMR, ESD or
NBO payload was frozen for later serving (see
[`docs/PHOTOPHYSICS.md`](PHOTOPHYSICS.md)); it is present only when that
count is greater than 0.

Refused with `409` for the protected `admin/default` project, for one
that is already archived, while a job or a wipe of the project is under
way (as for the exports), and for one with a flagged NMR molecule whose
reference compound is still `pending` — a frozen molecule is never
re-analysed, so the shift would stay `null` forever (a `failed`
reference does not block it: the frozen detail keeps the shieldings, the
shift stays `null`); `404` when the project holds no molecules. From
the moment the job starts, the controller creates and submits no job for
the project, not even later in a tick already under way. The archive job
then waits up to 300 s until every job the project sent to SLURM has
finished and been judged and its task marked, so a job that finishes
during the wait is exported, and until no cube build or task-file download
of the project is still open. A job whose `sbatch` reported failure may
still have been accepted: the controller looks it up in SLURM (it sends
no new `sbatch`), and the archive waits for it if found, or treats it as
never sent if not. If any are still running, unjudged or not looked up
after that, it aborts and deletes nothing. It then fails the project's `created`
and `pending` tasks with the reason `project archived`; they do not count
towards the circuit breaker.

---

## 4. Headers

Stored ORCA header templates that populate the dashboard's submission
dropdowns. Six are seeded on first init; the rest are admin's, or
were created by users before header writes became admin-only.

Every signed-in account may list and use headers; only admin creates,
edits or deletes one (see §0). The `defaults` block in
the listing below is separate again: three package constants, always
present, addressed by the string ids `default_confsearch` /
`default_optimization` / `default_singlepoint` and not editable.

### `GET /api/headers` `?kind=confsearch|optimization|singlepoint&include_deleted=true`

```json
{
  "defaults": [ { "id": "default_confsearch", "label": "...",
                  "description": "...", "kind": "confsearch",
                  "text": "!GOAT XTB\n%xtb\n  XTBInputString \"--gxtb\"\nend\n...",
                  "legacy": false } ],
  "custom":  [ { "id": 1, "label": "GOAT GFN2-xTB conformer ensemble",
                 "description": "...", "kind": "confsearch",
                 "validated": true, "deleted": false,
                 "text": "!GOAT XTB2\n...", "legacy": false } ]
}
```

Strict `kind` filter: untagged custom headers are intentionally hidden
from slot-specific responses; they're still listed when no `kind` is
passed. Soft-deleted headers are excluded unless `include_deleted=true`.
`legacy` marks a header a submission with Advanced settings refuses
(see `advanced` under `POST /api/submit`): for a confsearch header only
`PALn`, `%pal` and `%maxcore`, for the others anything Advanced owns.

### `POST /api/headers`

```json
{ "header_text": "!wB97X-D3 def2-TZVPD\n",
  "description": "wB97X-D3/def2-TZVPD singlepoint",
  "kind":        "singlepoint",
  "validated":   false }
```

`kind` must be one of `confsearch | optimization | singlepoint | null`;
`""` is stored as `null`.

Admin only (**403** otherwise).

### `PUT /api/headers/{id}`

Partial update — pass only the fields you want changed. Admin only
(**403** otherwise).

When any state or task (and so any job) references the header, a change of
`header_text` or `kind` creates a new header instead: the new text, kind and
description, unvalidated, owned by admin. The used header is left
untouched, so its states, retries, follow-ups and the Methods page keep the
text that ran. A `description` or `validated` change alone, or a body whose
text and kind equal the stored ones, edits in place. The answer is the
resulting header with `"copied": true` when a new one was made:

```json
{ "id": 12, "header_text": "!PBE0 def2-SVP\n", "description": "...",
  "kind": "optimization", "validated": false, "copied": true }
```

### `DELETE /api/headers/{id}`

Soft-delete. Sets `deleted=true`; the row stays in the table so finished
tasks keep their FK pointers. Admin only (**403** otherwise). Users can
no longer submit with it (**400**); work already on it keeps running.
Refused with `409` only when an **in-flight** task (`created` or
`pending`) still references the header, directly or through its state.

---

## 5. Failure handling

The pipeline is built to fail loudly:

* `POST /api/submit` returns **400** on invalid SMILES *before* anything
  is persisted.
* SMILES that pass the syntactic check but later break geometry
  generation set `processing_error` on the entrypoint and appear on
  `GET /api/entrypoints/failed`. The controller does not retry them.
* A job that runs but fails ORCA's checks sets `success=false` and a
  `fail_reason` on `ComputationJob`. The owning task moves to `failed`
  after `pipeline.max_attempts` unsuccessful jobs (default 3). A job also
  fails as an attempt after 3 consecutive sbatch rejections
  (`sbatch rejected 3 times: …`), in an unknown SLURM state
  (`Unrecognised SLURM state <state>`), or when SLURM has no record of it
  on two polls running (`SLURM has no record of the job`).
* A task fails at once when its pending job is unsatisfiable
  (`unsatisfiable: <Reason>`, e.g. `PartitionConfig`), when building its
  job raised 10 consecutive times, or with `project archived` when its
  project is archived. The README's "Failure handling" has the details.

To recover: fix the SMILES / header / config and resubmit. For the
recovery CLI commands see the README.

---

## 6. Destructive admin operations

Every one of these is irreversible and comes in two halves: a
`GET …/wipe-preview` (or `/api/admin/reset-preview`) that only counts, and
a `POST` that acts and requires `confirm` to echo an exact string.

| Endpoint | Who | Confirm with | Deletes |
|---|---|---|---|
| `GET /api/projects/{name}/wipe-preview` | owner or admin | — | nothing |
| `POST /api/projects/{name}/wipe` | owner or admin | the **qualified** project name | its rows, `comp_data/mol_*`, and (only with `delete_exports: true`) the export directory |
| `GET /api/molecules/{id}/wipe-preview` | owner or admin | — | nothing |
| `POST /api/molecules/{id}/wipe` | owner or admin | the molecule's SMILES | that molecule's rows and `comp_data/mol_{id}` |
| `GET /api/admin/reset-preview` | admin | — | nothing |
| `POST /api/admin/reset-database` | admin | `RESET THE DATABASE` | every pipeline table; data directories unless `delete_files: false`; headers only if `keep_headers: false` |
| `GET /api/wipe-status` | anyone | — | nothing; reports the deletion still in flight |
| `GET /api/admin/disk-usage` | admin | — | nothing; the last measurement, or `null` |
| `POST /api/admin/disk-usage` | admin | — | nothing; starts a measurement |

A user may wipe their own projects and molecules; someone else's answers
**404**, the same as a read, since a 403 would confirm it exists. The
project page's "Delete Project" button is this project wipe, offered to the
owner and admin; the Admin page keeps its own. Only the
shared `admin/default` is protected — your own `alice/default` is an
ordinary project. `wipe-status` tells a non-admin only *that* something is
running: the label names the project, which may not be theirs.

Saved headers are never touched by a project or molecule wipe. A wipe body
with any field besides `confirm` and `delete_exports` answers **422**.

Five properties worth knowing:

* **One at a time.** A destructive operation started while another is
  running is refused with **409**, not queued.
* **Rows first, files second.** The database rows are deleted and
  committed before anything is unlinked, so a failure mid-rmtree leaves
  orphaned directories — named in the log and in `orphaned_dirs` — rather
  than an emptied disk with rows still pointing at it.
* **The files go in the background.** Unlinking is ~65 ms per file on this
  deployment's network mount, so a real project is minutes. A project wipe
  and a database reset therefore *stage* — rename into a `.wipe-trash/`
  directory — and return immediately, leaving a background thread to
  unlink. A wipe stages one directory per molecule (~24 ms each); a reset
  stages `comp_data` and `export_data` whole, three syscalls no matter how
  much is under them. Measured on a 300-file tree: **0.14 s to stage
  against 9.8 s to delete**, and only the staging is inside the request.
  The response's `file_removal` object reports progress, and
  `GET /api/wipe-status` follows it to completion.
* **SLURM is stopped before the files go.** Jobs the scheduler still has
  queued or running are `scancel`ed after the rows are committed and before
  their directories disappear; the count comes back as `jobs_cancelled`.
  A `scancel` that has not answered after 60 s, or that fails, is given up:
  the files are deleted anyway and `scancel_warning` names the SLURM ids
  that may still run (`null` otherwise). A job SLURM may have but with no
  job id to cancel yet (claimed for `sbatch` this moment, or sent by an
  `sbatch` that reported failure) makes the wipe or reset answer **409**
  and delete nothing; retry once the controller has run again and recorded
  it, which a crash-left claim also waits for. The checks and the row
  deletes run under one database write lock, so no `sbatch` claim lands
  between them; `scancel` runs after it is released.
* **Not beside a project job.** A project or molecule wipe answers
  **409** while the project has an archive or export running (detail
  `Export and Delete is running for this project.` for the archive), and
  a reset while any project has one. Until the wipe's rows are gone, an
  export or archive of that project (any project, for a reset) answers
  **409**.

### Sizes, and why they are not free

Measuring a directory means one `stat` per file, and on this deployment's
network mount that is milliseconds each: 32 GB of `comp_data` in 58,473
files took **5m14s** to walk. So no request measures without a bound.

* A **wipe preview** measures under a 2-second budget and reports
  `files.measured_completely`. When that is `false` every byte figure in
  the response is a *lower* bound and the dialog says "at least" — better
  than under-reporting what is about to be deleted.
* `GET /api/admin/reset-preview` **reads no files at all**. It returns
  `projects`, `rows` and `confirmation_required`. The reset does not need
  file counts — it stages
  directories with a rename, so what is on disk does not change what it
  does.
* The exact figure is `POST /api/admin/disk-usage`, which starts a walk on
  its own thread and answers immediately. Poll `GET /api/admin/disk-usage`
  for `{"state": "running"|"ready"|"failed", "dirs_done", "dirs_total",
  "files", "comp_data_bytes", "export_bytes", "total_bytes",
  "elapsed_seconds", "measured_at"}`. A POST arriving while a walk is
  running joins it rather than starting a second one. The result is cached
  until a wipe or reset invalidates it.

This is not a micro-optimisation. `reset-preview` was fetched on every
render of the admin page, ran inside a `get_session()` block, and a
synchronous handler is not cancelled when the browser navigates away — so
each reload pinned one of the connection pool's slots for five minutes.
A handful of reloads exhausted the pool and every request in the process,
including submissions and API-key authentication, blocked behind it.

Staging is not just about latency. Molecule ids restart at 1 after a
reset and the worker runs in the same process as the API, so within a tick
of the reset returning it is creating `mol_1` again — into the very
directory a deleter would still have been walking. Renaming frees the name
before the response is sent, so the new tree and the doomed one can never
be the same tree.

Because the operation is not finished when the response arrives, another
wipe stays refused with 409 until the last file is gone. `wipe-status`
answers `{"running": ..., "operation": ..., "file_removal": {...}}`; the
dashboard shows a banner while one is in flight. A controller killed
mid-deletion leaves its batch under `.wipe-trash/`, which the next wipe or
reset sweeps. A project wipe stages its export folder under
`<export_data>/<owner>/.wipe-trash/<pid>-<n>/`, which no wipe sweeps; the
controller removes those batches (only ones whose process is gone) on a
background thread when it starts, logging each.

---

## 7. Cluster health

### `GET /api/cluster`

Readable by everyone who is signed in, and deliberately thin — it carries
no one's data:

```json
{ "breaker_tripped": false, "queued_entrypoints": 143 }
```

`breaker_tripped` is the caller's own breaker (see below); for admin it is
true while any owner's breaker is tripped. That is enough
to tell "my jobs are stuck" from "the pipeline is halted" without asking an
administrator.

### `GET /api/admin/circuit-breaker`

Admin only. The breaker is kept per project owner (the namespace before
`/` in the project name; names without one are `admin`'s). This lists every
owner that is tripped or has failures among their recent judged tasks:

```json
{ "tripped": true,
  "owners": [
    { "owner": "alice", "tripped": true,
      "state": { "owner": "alice", "tripped_at": "2026-10-28T10:00:00+00:00",
                 "ratio": 0.93, "failed": 93, "judged": 100, "detail": "…" },
      "recent_failure_ratio": 0.93, "recent_failed": 93, "recent_judged": 100 },
    { "owner": "bob", "tripped": false, "state": null,
      "recent_failure_ratio": 0.08, "recent_failed": 8, "recent_judged": 100 } ],
  "threshold": 0.9, "window": 100 }
```

`tripped` is true when any owner is. `threshold` and `window` echo
`pipeline.failure_breaker_ratio` and `pipeline.failure_breaker_window`. An
owner's job creation and submission stop automatically once more than
`threshold` of their last `window` judged tasks failed (with at least
`pipeline.failure_breaker_min_samples` judged), so one systematic error
cannot burn their whole campaign's retry budget. Their queued entrypoints
wait unexpanded. Other owners carry on, and work already submitted keeps
running.

### `POST /api/admin/circuit-breaker/reset`

Admin only. Body `{"owner": "alice"}` clears that owner's breaker; no body
(or `owner: null`) clears every owner's. Answers
`{"reset": ["alice"], "tripped": ["bob"]}`: the owners cleared and those
still tripped; `400` for an invalid owner name. Deliberately manual: once
submission stops, no new tasks are judged, so the failure ratio cannot
recover on its own. Fix the cause first. A reset starts the owner's failure
count afresh.

### `GET /api/admin/submission-hold`

Admin only. A manual pause of `sbatch` submission, independent of the
breaker: `{"held": false, "state": null}`, or `held: true` with
`state = {"held_at", "held_by", "reason"}`.

### `POST /api/admin/submission-hold`

Admin only. Body `{"reason": "…"}` (optional, ≤ 500 chars). Stops submitting
jobs; finished jobs are still parsed and followups prepared, so they queue
up until release.

### `POST /api/admin/submission-hold/release`

Admin only. Submission resumes on the next tick.
