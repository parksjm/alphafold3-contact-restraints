# Recovered production contact guidance: scientific regression

Date: 2026-10-13. This document supersedes the negative assessment of the
different azalea `bc20b9d` implementation. It does not erase those failed tests.

## Recovery and preservation

The successful production source was found, clean and intact, at
`violet:/data/jpc/code/alphafold3-contact-restraints`, HEAD
`ef5dc91d65b4845f63e95ea03316f1613c6dfb5e`. Azalea had a separate checkout that
diverged before that correction; the same pathname on two hosts did not mean
the same Git history. No evidence was found that the successful commit had
been deleted.

The missing history was fetched directly from violet. Preserved references on
azalea are:

- `codex/recovered-production-ef5dc91`: untouched production source;
- `codex/contact-restraint-guidance`: prior azalea `bc20b9d` safeguards variant;
- `codex/contact-restraint-recovery`: recovered production plus these safeguards.

An untouched detached baseline worktree is also available at
`/tmp/af3-production-ef5dc91`. Neither host's original
`/data/jpc/code/alphafold3` was edited. Violet's working contact fork was read
only: no checkout, package install, source edit, or new inference job there.
No dependency packages were installed into the existing AF3 environment.

## What was preserved and what changed

The scientific optimizer remains the one in `ef5dc91`: lower/upper flat-bottom
linear contacts, explicitly mobile whole chains, sequential corrections in
input order, separate 0.25-A translation and 2-degree rotation caps, and the
validated terminal-guidance schedule. No joint-gradient replacement, steric
potential, backtracking, covalent-bond workaround, or final coordinate
projection was introduced. The exact algorithm and limitations are documented
in [input.md](input.md#exact-scientific-behavior-of-the-recovered-implementation).

Added safeguards are observational checks or rejection of unsupported inputs:

- finite, positive-integer config validation, including configs mutated by
  runners after construction;
- endpoint/mobility masks, valid float32 bounds and weights, whole-chain masks,
  directly contradictory same-pair intervals, invariant intra-chain rigid
  restraints, and declared inter-chain covalent links;
- per-step valid-coordinate nonfinite counts at four sampler stages;
- full-precision failure when any sample misses any required contact;
- full-precision endpoint/bound/mobility/settings/trace provenance in
  `contact_restraints.json`;
- a tracing fix for legacy feature dictionaries with zero restraint slots;
- omission of the fork-only field from ordinary unrestrained input JSON;
- preservation of user-provided XLA flags in the Kuhlman runner.

Zero-weight runs remain explicit negative controls: contacts need not be
satisfied, but coordinates must be finite. Contact failure is not a seed/sample
selection policy: the run fails; samples are not filtered or repaired.

## Matched real-weight experiments

Exact processed inputs were copied from the canonical violet project; MSAs,
templates, SMILES, contact order and chain IDs were not regenerated or edited.
The one-copy input contains A/E/I, not a renumbered canonical full-length
protein. The historical `fourcopy` filename means four alpha/beta pairs plus
**one** ligand; the separate `four_site` input has four ligands and twelve
contacts. Alpha IDs are A/B/C/D; beta IDs are E/F/G/H. Beta Thr1 is canonical
Thr233; Phe113 and Asn186 remain alpha chain-local residues.

| System | Seed | Thr1 OG1--C12 | Phe113 N--O4 | Asn186 ND2--O4 |
| --- | ---: | ---: | ---: | ---: |
| A/E/I, guidance off | 2026100911 | 7.9326 | 5.8539 | 9.1837 |
| A/E/I, recovered defaults | 2026100911 | 2.6621 | 2.7414 | 3.0831 |
| Eight protein chains + I | 2026100911 | 2.5453 | 3.0810 | 2.5753 |
| A/E/I site, four ligands | 2026100921 | 2.5158 | 2.6678 | 3.4247 |
| B/F/J site, four ligands | 2026100921 | 2.5598 | 2.7238 | 2.7705 |
| C/G/K site, four ligands | 2026100921 | 2.5581 | 2.5129 | 3.6019 |
| D/H/L site, four ligands | 2026100921 | 2.5232 | 2.5469 | 3.6024 |

Distances are Angstroms. Bounds are 2.5--3.2 for Thr--C12 and 2.5--3.8 for
the two oxygen contacts. All guided cases pass every lower and upper bound.
The four-site case passes 12/12. The baseline and safeguarded runs have
**bit-for-bit identical final coordinates**, including the unguided reduced
control; maximum component-wise difference is 0 A. Same-host comparisons are
the regression evidence: small cross-host/backend differences from historical
violet distances are not claimed to be bitwise reproducible.

All guided cases pass the harness's deliberately coarse checks: no modeled
bond outside 0.9--2.2 A, no ligand bond outside that window, and no inter-chain
nonbonded atom pair below 1.5 A. These checks are not force-field validation,
bond-order-specific validation, a full stereochemical audit, or proof of a
catalytically productive pose. Required distances were also independently
measured by selecting literal chain/residue/atom IDs from emitted CIFs;
agreement within 0.002 A allows for CIF coordinate rounding.

Runtime: actual standard AF3 weights, azalea RTX PRO 6000 Blackwell GPUs,
JAX 0.6.0, cuDNN attention, float32 highest matmul precision, 10 recycles,
200 diffusion steps, one sample per seed. Guidance: weight 0.5, interval 1,
20 inner updates, final weight scale 1. The bucket is 512 for A/E/I and 1536
for both octamer cases. `LD_LIBRARY_PATH` was unset, GPU preallocation disabled,
and command-buffer capture disabled consistently for all comparisons.

Input SHA-256 values:

- A/E/I: `1ec08fa6c89925df8044da90611af1fde14f33a74ff7818a039054ff5b7051fc`.
- Octamer + I: `5b4966db1e6e1c8625767d9d0d270031be139e4239296ef43d52944bbc39b2f0`.
- Four sites: `14202fdb2fe5518e4fcbe5891b3fb662161af0505a079999c9a691fbea0f5c4c`.

## Evidence and reproduction

Inputs, both baseline and safeguarded reports, raw coordinate/trace NPZ files,
CIFs and confidence/provenance JSON are retained on azalea in the ignored
`contact_recovery_validation_20261013/` directory of this fork. In particular:

- `af3_recovered_safeguards_onecopy_final/report.json`;
- `af3_recovered_safeguards_fourcopy/report.json`;
- `af3_recovered_safeguards_four_site/report.json`.

The durable harness is `contact_scientific_smoke.py`. It uses the first seed
of a multi-seed input, records numerical metrics, retains raw coordinates and
diagnostics before extraction, and exits nonzero when a guided case misses a
bound or the coarse geometry screen. An import fallback exists only for the
untouched baseline, which predates the new host validation module.

From the recovered fork on azalea:

```bash
env -u LD_LIBRARY_PATH \
  PYTHONPATH="$PWD/src:/data/jpc/code/.conda/envs/af3/lib/python3.11/site-packages" \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  XLA_FLAGS='--xla_gpu_enable_command_buffer=' \
  /data/jpc/code/.conda/envs/af3/bin/python -S contact_scientific_smoke.py \
  --input contact_recovery_validation_20261013/inputs/nyl12_production_four_site.json \
  --model-dir /data/jpc/code/alphafold3/models \
  --output /tmp/af3_contact_fresh_four_site \
  --backend cudnn --weights 0.5 --bucket 1536
```

The output directory must not already exist. For the untouched baseline,
replace `$PWD/src` with `/tmp/af3-production-ef5dc91/src` after preparing its
runtime overlay. This comparison uses the identical harness and inputs.

All **26 focused tests passed**, as did focused Ruff correctness checks,
Python compilation and `git diff --check`. Tests cover recovered-optimizer and sampler coordinate equivalence,
rigid distances/chirality, the validated schedule/update count, all four Nyl12
site mappings, lower and upper bound failure, config validation, missing atoms,
contradictory pair bounds, partial mobile masks, covalent links, nonfinite
diagnostics, old feature batches, ordinary JSON and full featurization. They
can be run with pytest, or directly without installing pytest into AF3:

```bash
PYTHONPATH="$PWD/src" /data/jpc/code/.conda/envs/af3/bin/python - <<'PY'
import contact_restraints_test as tests
for name in sorted(vars(tests)):
    if name.startswith('test_'):
        getattr(tests, name)()
PY
```

This is a bounded regression on fixed WT inputs, not a guarantee for all seeds,
ligands, conflicting restraints or catalytic poses. No mutant campaign,
seed sweep, post-filtering workflow, force-field relaxation or MD was run as
part of recovery. Preserve production contact order; do not assume the
sequential optimizer is order invariant.
