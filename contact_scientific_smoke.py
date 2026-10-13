"""Run a reproducible same-seed contact-weight comparison with actual weights.

Set XLA_FLAGS before launching; this script does not override it. Outputs are
created in a new directory, including diagnostics even if extraction fails.
"""

import argparse
import dataclasses
import json
import hashlib
import importlib.util
import os
import pathlib
import time

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.data.tools import rdkit_utils

if importlib.util.find_spec('alphafold3.model.contact_restraints') is not None:
  from alphafold3.model import contact_restraints

  validation_module_present = True
else:
  # Allows the same harness to execute the untouched ef5dc91 baseline.
  import types

  validation_module_present = False
  contact_restraints = types.SimpleNamespace(
    settings_for_batch=lambda batch, config: config.as_dict(),
    validate_input_features=lambda batch, config: None,
    validate_inference_result=lambda result, *args: None,
  )
from alphafold3.model import feat_batch
from alphafold3.model import model
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.network import diffusion_head
from alphafold3.model.components import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree


def geometry_metrics(batch, positions, fold_input, ccd):
  """Bond lengths and severe inter-chain overlaps; these are screening metrics.

  Bond limits are deliberately coarse (0.9--2.2 A). They are not a substitute
  for a force field, bond-type-specific reference lengths, or PoseBusters.
  """
  layout = batch.convert_model_output.token_atoms_layout
  active = layout.atom_name.astype(bool)
  coords = positions[active]
  names = layout.atom_name[active]
  chains = layout.chain_id[active]
  residues = layout.res_id[active]
  res_names = layout.res_name[active]
  lookup = {
    (str(c), int(r), str(a)): i
    for i, (c, r, a) in enumerate(zip(chains, residues, names, strict=True))
  }
  bonds = set()
  for chain_id, res_id, res_name in set(zip(chains, residues, res_names)):
    entry = ccd.get(res_name)
    if entry:
      for a, b in zip(
        entry.get('_chem_comp_bond.atom_id_1', []),
        entry.get('_chem_comp_bond.atom_id_2', []),
        strict=True,
      ):
        i, j = (
          lookup.get((chain_id, int(res_id), a)),
          lookup.get((chain_id, int(res_id), b)),
        )
        if i is not None and j is not None:
          bonds.add(tuple(sorted((i, j))))
    # Peptide bonds are absent from individual-residue CCD definitions.
    i = lookup.get((chain_id, int(res_id), 'C'))
    j = lookup.get((chain_id, int(res_id) + 1, 'N'))
    if i is not None and j is not None:
      bonds.add(tuple(sorted((i, j))))
  for ligand in fold_input.ligands:
    if ligand.smiles:
      molecule = rdkit_utils.assign_atom_names_from_graph(
        Chem.AddHs(Chem.MolFromSmiles(ligand.smiles))
      )
      for bond in molecule.GetBonds():
        a = bond.GetBeginAtom().GetProp('atom_name')
        b = bond.GetEndAtom().GetProp('atom_name')
        i, j = lookup.get((ligand.id, 1, a)), lookup.get((ligand.id, 1, b))
        if i is not None and j is not None:
          bonds.add(tuple(sorted((i, j))))
  for a, b in fold_input.bonded_atom_pairs or ():
    if a in lookup and b in lookup:
      bonds.add(tuple(sorted((lookup[a], lookup[b]))))
  bond_array = np.asarray(sorted(bonds), dtype=int).reshape(-1, 2)
  distances = np.linalg.norm(
    coords[bond_array[:, 0]] - coords[bond_array[:, 1]], axis=-1
  )
  overlaps = cKDTree(coords).query_pairs(1.5, output_type='ndarray')
  overlap_count = sum(
    chains[i] != chains[j] and (int(i), int(j)) not in bonds for i, j in overlaps
  )
  ligand_mask = np.isin(chains, [c.id for c in fold_input.ligands])
  ligand_bonds = ligand_mask[bond_array].all(axis=1)
  return {
    'bond_count': len(bonds),
    'bonds_outside_0_9_to_2_2_A': int(((distances < 0.9) | (distances > 2.2)).sum()),
    'max_bond_length_A': float(distances.max()) if len(distances) else None,
    'ligand_bond_count': int(ligand_bonds.sum()),
    'ligand_bonds_outside_0_9_to_2_2_A': int(
      (((distances < 0.9) | (distances > 2.2)) & ligand_bonds).sum()
    ),
    'interchain_nonbonded_pairs_below_1_5_A': int(overlap_count),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', required=True)
  parser.add_argument('--model-dir', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--weights', type=float, nargs='+', default=[0.0, 0.5])
  parser.add_argument('--backend', choices=['xla', 'cudnn', 'triton'], default='xla')
  parser.add_argument('--recycles', type=int, default=10)
  parser.add_argument('--steps', type=int, default=200)
  parser.add_argument('--interval', type=int, default=1)
  parser.add_argument('--inner-steps', type=int, default=20)
  parser.add_argument('--final-weight-scale', type=float, default=1.0)
  parser.add_argument('--bucket', type=int, default=None)
  args = parser.parse_args()
  output = pathlib.Path(args.output)
  output.mkdir(parents=True, exist_ok=False)
  fold_input = folding_input.Input.from_json(pathlib.Path(args.input).read_text())
  fold_input = dataclasses.replace(fold_input, rng_seeds=[fold_input.rng_seeds[0]])
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  (example,) = featurisation.featurise_input(
    fold_input, ccd, buckets=(args.bucket,) if args.bucket else None
  )
  batch = feat_batch.Batch.from_data_dict(example)
  arrays = jax.tree.map(jnp.asarray, utils.remove_invalidly_typed_feats(example))
  model_params = params.get_model_haiku_params(pathlib.Path(args.model_dir))
  seed = fold_input.rng_seeds[0]
  runs = []
  for weight in args.weights:
    config = model.Model.Config()
    config.global_config.flash_attention_implementation = args.backend
    config.num_recycles = args.recycles
    config.heads.diffusion.eval.num_samples = 1
    config.heads.diffusion.eval.steps = args.steps
    guidance = config.heads.diffusion.eval.contact_guidance
    guidance.guidance_weight = weight
    guidance.guidance_interval = args.interval
    guidance.final_weight_scale = args.final_weight_scale
    guidance.num_gradient_steps = args.inner_steps
    guidance.__post_init__()
    contact_restraints.validate_input_features(example, guidance)

    @hk.transform
    def forward(features):
      return model.Model(config)(features)

    started = time.monotonic()
    run = {
      'weight': weight,
      'seed': seed,
      'backend': args.backend,
      'guidance': contact_restraints.settings_for_batch(example, guidance),
      'jax_version': jax.__version__,
      'recycles': args.recycles,
      'diffusion_steps': args.steps,
      'bucket': args.bucket,
      'input_sha256': hashlib.sha256(pathlib.Path(args.input).read_bytes()).hexdigest(),
      'xla_flags': os.environ.get('XLA_FLAGS', ''),
      'matmul_precision': str(jax.config.jax_default_matmul_precision),
      'devices': [str(device) for device in jax.devices()],
      'source_path': diffusion_head.__file__,
      'source_sha256': hashlib.sha256(
        pathlib.Path(diffusion_head.__file__).read_bytes()
      ).hexdigest(),
      'validation_module_present': validation_module_present,
      'chain_ids': [chain.id for chain in fold_input.chains],
      'restraints': [
        dataclasses.asdict(c) for c in fold_input.contact_constraints or ()
      ],
    }
    prefix = output / f'weight_{weight:g}'
    print(f'Running {run}', flush=True)
    try:
      result = jax.jit(forward.apply)(model_params, jax.random.PRNGKey(seed), arrays)
      result = jax.tree.map(np.asarray, result)
      result = dict(result)
      result['contact_guidance_config'] = contact_restraints.settings_for_batch(
        example, guidance
      )
      result['__identifier__'] = model_params['__meta__']['__identifier__'].tobytes()
      samples = result['diffusion_samples']
      diagnostics = {
        key: value for key, value in samples.items() if key.startswith('contact_')
      }
      np.savez_compressed(
        str(prefix) + '_diagnostics.npz',
        **diagnostics,
        atom_positions=samples['atom_positions'],
      )
      coords = np.asarray(samples['atom_positions'])
      valid = np.asarray(samples['mask'], bool)[..., None]
      if np.any(~np.isfinite(coords) & valid):
        raise FloatingPointError('Nonfinite final valid atom coordinates.')
      active = np.asarray(batch.contact_constraints.mask, bool)
      run['distances_A'] = result['contact_restraint_distances'][0, active].tolist()
      run['thresholds_A'] = np.asarray(batch.contact_constraints.max_distances)[
        active
      ].tolist()
      run['min_thresholds_A'] = np.asarray(batch.contact_constraints.min_distances)[
        active
      ].tolist()
      run['final_restraint_energy'] = float(
        np.sum(
          (
            np.maximum(
              np.asarray(run['distances_A']) - np.asarray(run['thresholds_A']), 0.0
            )
            + np.maximum(
              np.asarray(run['min_thresholds_A']) - np.asarray(run['distances_A']), 0.0
            )
          )
          * np.asarray(batch.contact_constraints.weights)[active]
        )
      )
      run['geometry'] = geometry_metrics(
        batch, samples['atom_positions'][0], fold_input, ccd
      )
      run['restraints_satisfied'] = (
        (np.asarray(run['distances_A']) <= np.asarray(run['thresholds_A']))
        & (np.asarray(run['distances_A']) >= np.asarray(run['min_thresholds_A']))
      ).tolist()
      run['scientific_screen_passed'] = (
        all(run['restraints_satisfied'])
        and run['geometry']['bonds_outside_0_9_to_2_2_A'] == 0
        and run['geometry']['interchain_nonbonded_pairs_below_1_5_A'] == 0
      )
      contact_restraints.validate_inference_result(result, example, guidance)
      inference = list(
        model.Model.get_inference_result(example, result, fold_input.name)
      )
      prefix.mkdir()
      for sample in inference:
        post_processing.write_output(sample, prefix)
      run['status'] = 'complete'
    except Exception as error:
      run['status'] = 'failed'
      run['error'] = f'{type(error).__name__}: {error}'
      print(run['error'], flush=True)
    run['elapsed_seconds'] = time.monotonic() - started
    runs.append(run)
    (output / 'report.json').write_text(json.dumps(runs, indent=2, allow_nan=False))
    print(json.dumps(run, indent=2), flush=True)
  if any(
    run['status'] != 'complete'
    or (run['weight'] > 0 and not run.get('scientific_screen_passed', False))
    for run in runs
  ):
    raise SystemExit(1)


if __name__ == '__main__':
  main()
