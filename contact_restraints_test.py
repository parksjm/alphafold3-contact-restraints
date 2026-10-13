# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0.

"""Focused tests for inference-time contact-restraint guidance."""

import json
import types
import ast
import pathlib
import subprocess

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import confidence_types
from alphafold3.model import features
from alphafold3.model import contact_restraints
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.network import diffusion_head
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _fold_input_json(contact_constraints):
  return json.dumps({
      'name': 'contact_test',
      'modelSeeds': [1],
      'sequences': [
          {'protein': {'id': 'A', 'sequence': 'AA'}},
          {'protein': {'id': 'B', 'sequence': 'AA'}},
      ],
      'contactConstraints': contact_constraints,
      'dialect': folding_input.JSON_DIALECT,
      'version': folding_input.JSON_VERSION,
  })


def test_contact_constraint_json_round_trip():
  fold_input = folding_input.Input.from_json(
      _fold_input_json([
          {
              'token1': ['A', 1, 'CA'],
              'token2': ['B', 2, 'CA'],
              'minDistance': 2.5,
              'maxDistance': 7.5,
              'weight': 1.25,
              'mobileEndpoint': 'token2',
          }
      ])
  )

  assert fold_input.contact_constraints == (
      folding_input.ContactConstraint(
          token1=('A', 1, 'CA'),
          token2=('B', 2, 'CA'),
          min_distance=2.5,
          max_distance=7.5,
          weight=1.25,
          mobile_endpoint='token2',
      ),
  )
  round_tripped = folding_input.Input.from_json(fold_input.to_json())
  assert round_tripped.contact_constraints == fold_input.contact_constraints


def test_contact_constraint_rejects_invalid_residue():
  try:
    folding_input.Input.from_json(
        _fold_input_json([
            {'token1': ['A', 3], 'token2': ['B', 1], 'maxDistance': 6.0}
        ])
    )
  except ValueError as error:
    assert 'Invalid residue ID' in str(error)
  else:
    raise AssertionError('Invalid contact endpoint was accepted.')


def test_contact_guidance_reduces_violation():
  constraints = features.ContactConstraints(
      token_indices=jnp.array([[0, 1]], dtype=jnp.int32),
      atom_masks=jnp.array([[[True], [True]]]),
      min_distances=jnp.array([0.0], dtype=jnp.float32),
      max_distances=jnp.array([6.0], dtype=jnp.float32),
      weights=jnp.array([1.0], dtype=jnp.float32),
      mobile_endpoints=jnp.array([0], dtype=jnp.int32),
      mobile_atom_masks=jnp.zeros((1, 2, 1), dtype=bool),
      mask=jnp.array([True]),
  )
  positions = jnp.array([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])

  guided = diffusion_head.apply_contact_guidance(
      positions,
      constraints,
      guidance_weight=jnp.array(0.5),
      guidance_interval=4,
      num_gradient_steps=20,
  )
  before = diffusion_head.contact_restraint_min_distances(
      positions, constraints
  )
  after = diffusion_head.contact_restraint_min_distances(guided, constraints)

  np.testing.assert_array_less(np.asarray(after), np.asarray(before))
  np.testing.assert_array_less(np.asarray(after), np.array([6.01]))


def test_contact_guidance_interval_matches_boltz_update_count():
  constraints = features.ContactConstraints(
      token_indices=jnp.array([[0, 1]], dtype=jnp.int32),
      atom_masks=jnp.array([[[True], [True]]]),
      min_distances=jnp.array([0.0], dtype=jnp.float32),
      max_distances=jnp.array([1.0], dtype=jnp.float32),
      weights=jnp.array([1.0], dtype=jnp.float32),
      mobile_endpoints=jnp.array([0], dtype=jnp.int32),
      mobile_atom_masks=jnp.zeros((1, 2, 1), dtype=bool),
      mask=jnp.array([True]),
  )
  positions = jnp.array([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]])

  guided = diffusion_head.apply_contact_guidance(
      positions,
      constraints,
      guidance_weight=jnp.array(0.1),
      guidance_interval=4,
      num_gradient_steps=20,
  )

  # Updates occur at inner iterations 0, 4, 8, 12, and 16. Each update moves
  # both endpoint atoms by 0.1 A, reducing their separation by 0.2 A.
  distance = diffusion_head.contact_restraint_min_distances(
      guided, constraints
  )
  np.testing.assert_allclose(np.asarray(distance), [9.0], atol=1e-4)


def test_contact_feature_mapping_handles_residue_and_atom_endpoints():
  all_tokens = atom_layout.AtomLayout(
      atom_name=np.array(['CA', 'CA'], dtype=object),
      res_id=np.array([1, 2]),
      chain_id=np.array(['A', 'B'], dtype=object),
  )
  all_token_atoms = atom_layout.AtomLayout(
      atom_name=np.array(
          [['N', 'CA', 'C'], ['N', 'CA', 'C']], dtype=object
      ),
      res_id=np.array([[1, 1, 1], [2, 2, 2]]),
      chain_id=np.array([['A', 'A', 'A'], ['B', 'B', 'B']], dtype=object),
  )
  constraint = folding_input.ContactConstraint(
      token1=('A', 1), token2=('B', 2, 'CA'), max_distance=7.0
  )

  mapped = features.ContactConstraints.compute_features(
      [constraint],
      all_tokens,
      all_token_atoms,
      num_tokens=2,
      max_constraints=2,
  )

  np.testing.assert_array_equal(mapped.token_indices[0], [0, 1])
  np.testing.assert_array_equal(mapped.atom_masks[0, 0], [True, True, True])
  np.testing.assert_array_equal(mapped.atom_masks[0, 1], [False, True, False])
  np.testing.assert_array_equal(mapped.min_distances, [0.0, 0.0])
  np.testing.assert_array_equal(mapped.mask, [True, False])


def test_rigid_mobile_endpoint_preserves_internal_distance():
  constraints = features.ContactConstraints(
      token_indices=jnp.array([[0, 1]], dtype=jnp.int32),
      atom_masks=jnp.array([[[True], [True]]]),
      min_distances=jnp.array([0.0], dtype=jnp.float32),
      max_distances=jnp.array([0.1], dtype=jnp.float32),
      weights=jnp.array([1.0], dtype=jnp.float32),
      mobile_endpoints=jnp.array([2], dtype=jnp.int32),
      mobile_atom_masks=jnp.array([[[False], [True], [True]]]),
      mask=jnp.array([True]),
  )
  positions = jnp.array([
      [[0.0, 1.0, 0.0]],
      [[1.0, 0.0, 0.0]],
      [[0.0, 0.0, 0.0]],
  ])

  guided = diffusion_head.apply_contact_guidance(
      positions,
      constraints,
      guidance_weight=jnp.array(0.1),
      guidance_interval=1,
      num_gradient_steps=1,
  )

  before = diffusion_head.contact_restraint_min_distances(
      positions, constraints
  )
  after = diffusion_head.contact_restraint_min_distances(guided, constraints)
  np.testing.assert_array_less(np.asarray(after), np.asarray(before))
  np.testing.assert_allclose(
      np.linalg.norm(np.asarray(guided[1, 0] - guided[2, 0])),
      np.linalg.norm(np.asarray(positions[1, 0] - positions[2, 0])),
      atol=1e-5,
  )


def test_rigid_mobile_endpoint_respects_lower_bound():
  constraints = features.ContactConstraints(
      token_indices=jnp.array([[0, 1]], dtype=jnp.int32),
      atom_masks=jnp.array([[[True], [True]]]),
      min_distances=jnp.array([2.5], dtype=jnp.float32),
      max_distances=jnp.array([3.2], dtype=jnp.float32),
      weights=jnp.array([1.0], dtype=jnp.float32),
      mobile_endpoints=jnp.array([2], dtype=jnp.int32),
      mobile_atom_masks=jnp.array([[[False], [True], [True]]]),
      mask=jnp.array([True]),
  )
  positions = jnp.array([
      [[0.0, 0.0, 0.0]],
      [[0.9, 0.0, 0.0]],
      [[2.9, 0.0, 0.0]],
  ])

  guided = diffusion_head.apply_contact_guidance(
      positions,
      constraints,
      guidance_weight=jnp.array(0.5),
      guidance_interval=1,
      num_gradient_steps=20,
  )

  distance = diffusion_head.contact_restraint_min_distances(
      guided, constraints
  )
  np.testing.assert_array_less(np.array([2.49]), np.asarray(distance))


def test_old_feature_batches_get_empty_contact_constraints():
  constraints = features.ContactConstraints.from_data_dict({
      'pred_dense_atom_mask': np.ones((4, 3), dtype=bool)
  })
  assert constraints.token_indices.shape == (0, 2)
  assert constraints.atom_masks.shape == (0, 2, 3)


def test_sampler_runs_with_contact_guidance():
  constraints = features.ContactConstraints(
      token_indices=jnp.array([[0, 1]], dtype=jnp.int32),
      atom_masks=jnp.array([[[True], [True]]]),
      min_distances=jnp.array([0.0], dtype=jnp.float32),
      max_distances=jnp.array([6.0], dtype=jnp.float32),
      weights=jnp.array([1.0], dtype=jnp.float32),
      mobile_endpoints=jnp.array([0], dtype=jnp.int32),
      mobile_atom_masks=jnp.zeros((1, 2, 1), dtype=bool),
      mask=jnp.array([True]),
  )
  batch = types.SimpleNamespace(
      predicted_structure_info=types.SimpleNamespace(
          atom_mask=jnp.ones((2, 1), dtype=bool)
      ),
      contact_constraints=constraints,
  )
  config = diffusion_head.SampleConfig(
      steps=4,
      num_samples=1,
      contact_guidance={
          'guidance_weight': 0.5,
          'guidance_interval': 1,
          'num_gradient_steps': 2,
      },
  )

  def forward():
    return diffusion_head.sample(
        denoising_step=lambda positions, _: positions * 0.5,
        batch=batch,
        key=hk.next_rng_key(),
        config=config,
    )

  transformed = hk.transform(forward)
  key = jax.random.PRNGKey(0)
  params = transformed.init(key)
  result = transformed.apply(params, key)

  assert result['atom_positions'].shape == (1, 2, 1, 3)
  assert np.isfinite(np.asarray(result['atom_positions'])).all()
  assert result['contact_restraint_distances_after_step'].shape == (4, 1, 1)
  np.testing.assert_allclose(
      np.asarray(result['contact_guidance_weights'])[:, 0],
      [0.5, 0.25, 0.25, 0.5],
  )


def test_restraint_diagnostics_are_written_to_confidence_json():
  confidence = confidence_types.StructureConfidenceFull(
      pae=np.zeros((1, 1)),
      token_chain_ids=['A'],
      token_res_ids=[1],
      atom_plddts=[90.0],
      atom_chain_ids=['A'],
      contact_probs=np.ones((1, 1)),
      contact_restraint_distances=[5.5],
      contact_restraint_min_thresholds=[2.5],
      contact_restraint_thresholds=[6.0],
      contact_restraint_satisfied=[True],
      contact_restraint_distances_before_guidance=[[9.0]],
      contact_restraint_distances_after_guidance=[[5.5]],
      contact_restraint_distances_after_step=[[5.7]],
      contact_guidance_weights=[0.5],
  )
  encoded = json.loads(confidence.to_json())
  assert encoded['contact_restraint_distances'] == [5.5]
  assert encoded['contact_restraint_min_thresholds'] == [2.5]
  assert encoded['contact_restraint_thresholds'] == [6.0]
  assert encoded['contact_restraint_satisfied'] == [True]


def test_contact_constraint_survives_full_featurisation():
  raw_input = json.loads(
      _fold_input_json([
          {'token1': ['A', 1], 'token2': ['B', 2], 'maxDistance': 7.0}
      ])
  )
  for sequence in raw_input['sequences']:
    sequence['protein'].update({
        'unpairedMsa': '',
        'pairedMsa': '',
        'templates': [],
    })
  fold_input = folding_input.Input.from_json(json.dumps(raw_input))

  (batch,) = featurisation.featurise_input(
      fold_input=fold_input,
      ccd=chemical_components.cached_ccd(),
      buckets=(8,),
  )

  np.testing.assert_array_equal(batch['contact_constraint_mask'][:2], [True, False])
  np.testing.assert_array_equal(
      batch['contact_constraint_token_indices'][0], [0, 3]
  )
  assert batch['contact_constraint_max_distances'][0] == 7.0


def _safety_batch():
  constraints = features.ContactConstraints(
      token_indices=np.array([[0, 1]], np.int32),
      atom_masks=np.ones((1, 2, 1), bool),
      min_distances=np.array([2.5], np.float32),
      max_distances=np.array([3.2], np.float32),
      weights=np.array([1.0], np.float32),
      mobile_endpoints=np.array([2], np.int32),
      mobile_atom_masks=np.array([[[False], [True], [True], [True], [True]]]),
      mask=np.array([True]),
  )
  return constraints.as_data_dict() | {'pred_dense_atom_mask': np.ones((5, 1), bool)}


def _assert_raises(kind, function, text):
  try:
    function()
  except kind as error:
    assert text in str(error), str(error)
  else:
    raise AssertionError(f'Expected {kind.__name__}: {text}')


def test_invalid_and_mutated_guidance_configs():
  for name, value in [('guidance_weight', np.nan), ('final_weight_scale', np.inf),
                      ('guidance_interval', 1.5), ('num_gradient_steps', True)]:
    _assert_raises(ValueError, lambda: diffusion_head.ContactGuidanceConfig(**{name: value}), name)
    config = diffusion_head.ContactGuidanceConfig()
    setattr(config, name, value)
    _assert_raises(ValueError, lambda: contact_restraints.validate_input_features({}, config), name)


def test_float32_invalid_bounds_and_missing_atoms():
  config = diffusion_head.ContactGuidanceConfig()
  for key, value in [('contact_constraint_max_distances', np.inf),
                     ('contact_constraint_weights', 0.),
                     ('contact_constraint_min_distances', 3.2)]:
    batch = _safety_batch()
    batch[key][:] = value
    _assert_raises(ValueError, lambda: contact_restraints.validate_input_features(batch, config), 'float32')
  batch = _safety_batch()
  batch['contact_constraint_atom_masks'][0, 0] = False
  _assert_raises(ValueError, lambda: contact_restraints.validate_input_features(batch, config), 'valid predicted atoms')


def test_rigid_feature_mobility_validation():
  config = diffusion_head.ContactGuidanceConfig()
  batch = _safety_batch()
  contact_restraints.validate_input_features(batch, config)
  batch['contact_constraint_mobile_atom_masks'][0, 0] = True
  _assert_raises(ValueError, lambda: contact_restraints.validate_input_features(batch, config), 'both endpoints')


def test_conflicting_bounds_are_rejected():
  batch = _safety_batch()
  for key in list(batch):
    if key.startswith('contact_constraint_'):
      batch[key] = np.concatenate([batch[key], batch[key]], axis=0)
  batch['contact_constraint_min_distances'][1] = 4.
  batch['contact_constraint_max_distances'][1] = 5.
  # Reversed endpoints still refer to the same physical distance.
  batch['contact_constraint_token_indices'][1] = [1, 0]
  batch['contact_constraint_mobile_endpoints'][1] = 1
  _assert_raises(ValueError, lambda: contact_restraints.validate_input_features(
      batch, diffusion_head.ContactGuidanceConfig()), 'Conflicting bounds')


def test_required_contacts_fail_at_full_precision_but_zero_weight_is_control():
  result = {'diffusion_samples': {'atom_positions': np.zeros((1, 5, 1, 3)),
                                  'mask': np.ones((1, 5, 1), bool)},
            'contact_restraint_distances': np.array([[3.20001]], np.float32)}
  batch = _safety_batch()
  config = diffusion_head.ContactGuidanceConfig()
  _assert_raises(ValueError, lambda: contact_restraints.validate_inference_result(
      result, batch, config), 'unsatisfied')
  config.guidance_weight = 0.
  contact_restraints.validate_inference_result(result, batch, config)
  result['contact_restraint_distances'][:] = 2.49
  config.guidance_weight = 0.5
  _assert_raises(ValueError, lambda: contact_restraints.validate_inference_result(
      result, batch, config), 'unsatisfied')


def test_nonfinite_diagnostics_report_step_sample_and_stage():
  result = {'diffusion_samples': {'atom_positions': np.zeros((2, 5, 1, 3)),
                                  'mask': np.ones((2, 5, 1), bool),
                                  'contact_guidance_nonfinite_counts': np.zeros((4, 2, 4), int)}}
  result['diffusion_samples']['contact_guidance_nonfinite_counts'][2, 1, 2] = 1
  _assert_raises(FloatingPointError, lambda: contact_restraints.validate_inference_result(result),
                 'step=2, sample=1, stage=after guidance')


def test_covalently_linked_mobile_chain_is_rejected():
  raw = json.loads(_fold_input_json([{
      'token1': ['A', 1, 'CA'], 'token2': ['B', 1, 'CA'], 'mobileEndpoint': 'token2'}]))
  raw['bondedAtomPairs'] = [[['A', 2, 'C'], ['B', 1, 'N']]]
  _assert_raises(ValueError, lambda: folding_input.Input.from_json(json.dumps(raw)), 'covalent bond')


def _recovered_functions():
  repo = pathlib.Path(__file__).resolve().parent
  source = subprocess.check_output([
      'git', '-C', str(repo), 'show',
      'ef5dc91:src/alphafold3/model/network/diffusion_head.py'], text=True)
  namespace = dict(vars(diffusion_head))
  names = {'_clip_vector_norm', '_rotate_about_center',
           'apply_rigid_chain_contact_guidance', 'apply_contact_guidance',
           'contact_guidance_weight', 'sample'}
  for node in ast.parse(source).body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
      exec(compile(ast.unparse(node), '<recovered ef5dc91>', 'exec'), namespace)
  return namespace


def test_recovered_rigid_update_preserves_coordinates_chirality_and_order():
  # Exercise interacting contacts on the same noncoplanar rigid ligand.
  batch = _safety_batch()
  for key in list(batch):
    if key.startswith('contact_constraint_'):
      batch[key] = np.concatenate([batch[key], batch[key]], axis=0)
  batch['contact_constraint_token_indices'][1] = [0, 2]
  constraints = jax.tree.map(jnp.asarray, features.ContactConstraints.from_data_dict(batch))
  positions = jnp.array([[[0., 0., 0.]], [[8., 0., 0.]], [[9., 0., 0.]],
                         [[8., 1., 0.]], [[8., 0., 1.]]])
  kwargs = dict(guidance_weight=jnp.array(0.5), guidance_interval=1, num_gradient_steps=20)
  expected = _recovered_functions()['apply_contact_guidance'](positions, constraints, **kwargs)
  actual = diffusion_head.apply_contact_guidance(positions, constraints, **kwargs)
  np.testing.assert_array_equal(actual, expected)
  np.testing.assert_array_equal(actual[0], positions[0])
  before, after = np.asarray(positions[1:, 0]), np.asarray(actual[1:, 0])
  np.testing.assert_allclose(np.linalg.norm(before[:, None] - before[None], axis=-1),
                             np.linalg.norm(after[:, None] - after[None], axis=-1), atol=3e-6)
  np.testing.assert_allclose(np.linalg.det(before[1:] - before[0]),
                             np.linalg.det(after[1:] - after[0]), atol=3e-6)


def test_sampler_coordinates_match_recovered_with_and_without_guidance():
  constraints = jax.tree.map(jnp.asarray, features.ContactConstraints.from_data_dict(_safety_batch()))
  batch = types.SimpleNamespace(
      predicted_structure_info=types.SimpleNamespace(atom_mask=jnp.ones((5, 1), bool)),
      contact_constraints=constraints)
  for weight in (0., 0.5):
    config = diffusion_head.SampleConfig(steps=4, num_samples=1,
                                        contact_guidance={'guidance_weight': weight})
    def run(sampler):
      transformed = hk.transform(lambda: sampler(
          lambda x, _: x * 0.5, batch, hk.next_rng_key(), config))
      key = jax.random.PRNGKey(19)
      return transformed.apply(transformed.init(key), key)
    expected = run(_recovered_functions()['sample'])
    actual = run(diffusion_head.sample)
    np.testing.assert_array_equal(actual['atom_positions'], expected['atom_positions'])
    assert actual['contact_guidance_nonfinite_counts'].shape == (4, 1, 4)


def test_legacy_empty_restraints_can_be_sampled():
  constraints = features.ContactConstraints.from_data_dict({
      'pred_dense_atom_mask': np.ones((2, 1), bool)})
  batch = types.SimpleNamespace(
      predicted_structure_info=types.SimpleNamespace(atom_mask=jnp.ones((2, 1), bool)),
      contact_constraints=jax.tree.map(jnp.asarray, constraints))
  transformed = hk.transform(lambda: diffusion_head.sample(
      lambda x, _: x * 0.5, batch, hk.next_rng_key(),
      diffusion_head.SampleConfig(steps=4, num_samples=1)))
  key = jax.random.PRNGKey(20)
  result = transformed.apply(transformed.init(key), key)
  assert np.isfinite(result['atom_positions']).all()
  assert result['contact_restraint_distances_after_step'].shape == (4, 1, 0)


def test_nonfinite_padding_is_ignored_but_valid_atoms_fail():
  coords = np.zeros((1, 2, 1, 3))
  coords[0, 1] = np.nan
  result = {'diffusion_samples': {'atom_positions': coords,
                                 'mask': np.array([[[True], [False]]])}}
  contact_restraints.validate_inference_result(result)
  result['diffusion_samples']['mask'][0, 1] = True
  _assert_raises(FloatingPointError, lambda: contact_restraints.validate_inference_result(result),
                 'Nonfinite final valid atom')


def test_same_chain_rigid_constraint_is_rejected():
  raw = json.loads(_fold_input_json([{
      'token1': ['A', 1, 'CA'], 'token2': ['A', 2, 'CA'], 'mobileEndpoint': 'token2'}]))
  _assert_raises(ValueError, lambda: folding_input.Input.from_json(json.dumps(raw)), 'intra-chain')


def test_partial_mobile_chain_mask_is_rejected():
  batch = _safety_batch()
  batch['asym_id'] = np.array([1, 2, 2, 2, 2])
  config = diffusion_head.ContactGuidanceConfig()
  contact_restraints.validate_input_features(batch, config)
  batch['contact_constraint_mobile_atom_masks'][0, 4] = False
  _assert_raises(ValueError, lambda: contact_restraints.validate_input_features(batch, config), 'complete endpoint chain')


def test_unrestrained_json_omits_fork_only_field():
  raw = json.loads(_fold_input_json([]))
  del raw['contactConstraints']
  fold_input = folding_input.Input.from_json(json.dumps(raw))
  assert 'contactConstraints' not in json.loads(fold_input.to_json())


def test_nyl12_all_four_sites_use_chain_local_beta_thr1_and_alpha_residues():
  names, chains, residues, restraints = [], [], [], []
  for alpha, beta, ligand in [('A', 'E', 'I'), ('B', 'F', 'J'),
                               ('C', 'G', 'K'), ('D', 'H', 'L')]:
    chains.extend([beta, alpha, alpha, ligand, ligand])
    residues.extend([1, 113, 186, 1, 1])
    names.extend(['OG1', 'N', 'ND2', 'C12', 'O4'])
    for endpoint, atom, bound in [((beta, 1, 'OG1'), 'C12', 3.2),
                                   ((alpha, 113, 'N'), 'O4', 3.8),
                                   ((alpha, 186, 'ND2'), 'O4', 3.8)]:
      restraints.append(folding_input.ContactConstraint(
          token1=endpoint, token2=(ligand, 1, atom), min_distance=2.5,
          max_distance=bound, mobile_endpoint='token2'))
  tokens = atom_layout.AtomLayout(atom_name=np.array(names, object),
                                  chain_id=np.array(chains, object), res_id=np.array(residues))
  atoms = atom_layout.AtomLayout(atom_name=np.array(names, object)[:, None],
                                 chain_id=np.array(chains, object)[:, None], res_id=np.array(residues)[:, None])
  mapped = features.ContactConstraints.compute_features(restraints, tokens, atoms,
                                                        num_tokens=24, max_constraints=64)
  for site in range(4):
    np.testing.assert_array_equal(mapped.token_indices[3 * site:3 * site + 3],
                                   np.array([[0, 3], [1, 4], [2, 4]]) + 5 * site)
    assert mapped.mobile_atom_masks[3 * site].sum() == 2
  assert mapped.mask.sum() == 12
