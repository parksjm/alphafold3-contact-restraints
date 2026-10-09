# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0.

"""Focused tests for inference-time contact-restraint guidance."""

import json
import types

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data import featurisation
from alphafold3.model import confidence_types
from alphafold3.model import features
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
