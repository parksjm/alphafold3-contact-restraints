"""Host-side safeguards; these checks do not change the production optimizer."""

import numpy as np


def settings_for_batch(batch, config):
  del batch
  return dict(config.as_dict())


def validate_input_features(batch, config):
  """Revalidate mutated configs and active float32 restraint features."""
  config.__post_init__()
  active = np.asarray(batch.get('contact_constraint_mask', []), dtype=bool)
  if not active.any() or config.guidance_weight == 0:
    return
  from alphafold3.model import features

  batch = dict(batch) | features.ContactConstraints.from_data_dict(batch).as_data_dict()
  lower = np.asarray(
    batch.get('contact_constraint_min_distances', np.zeros(active.shape))
  )[active]
  upper = np.asarray(batch['contact_constraint_max_distances'])[active]
  weights = np.asarray(batch['contact_constraint_weights'])[active]
  if not (
    np.all(np.isfinite(lower) & (lower >= 0))
    and np.all(np.isfinite(upper) & (upper > lower))
    and np.all(np.isfinite(weights) & (weights > 0))
  ):
    raise ValueError('Active contact bounds/weights must remain valid in float32.')
  token_indices = np.asarray(batch['contact_constraint_token_indices'])[active]
  atom_masks = np.asarray(batch['contact_constraint_atom_masks'])[active]
  mobile = np.asarray(batch['contact_constraint_mobile_endpoints'])[active]
  mobile_masks = np.asarray(batch['contact_constraint_mobile_atom_masks'])[active]
  atom_mask = np.asarray(batch['pred_dense_atom_mask'], bool)
  if np.any((token_indices < 0) | (token_indices >= len(atom_mask))):
    raise ValueError('Contact token index is outside the feature batch.')
  if np.any(~atom_masks.any(axis=-1)) or np.any(atom_masks & ~atom_mask[token_indices]):
    raise ValueError('Contact endpoints must select valid predicted atoms.')
  if np.any(~np.isin(mobile, [0, 1, 2])):
    raise ValueError('Invalid mobile endpoint feature.')
  rigid = mobile > 0
  if np.any(atom_masks.sum(axis=-1)[rigid] != 1):
    raise ValueError('Rigid guidance requires exactly one atom at each endpoint.')
  if np.any(mobile_masks & ~atom_mask[None]):
    raise ValueError('Mobile chain mask includes padded atoms.')
  intersections = {}
  for i, (tokens, masks, endpoint) in enumerate(zip(token_indices, atom_masks, mobile)):
    if endpoint:
      side = int(endpoint) - 1
      if not mobile_masks[i].any() or not np.all(
        mobile_masks[i, tokens[side]][masks[side]]
      ):
        raise ValueError('Mobile endpoint atom is absent from its mobile chain mask.')
      if np.any(mobile_masks[i, tokens[1 - side]][masks[1 - side]]):
        raise ValueError('Rigid guidance cannot move both endpoints together.')
      if 'asym_id' in batch:
        asym_id = np.asarray(batch['asym_id'])
        whole_chain = (asym_id == asym_id[tokens[side]])[:, None] & atom_mask
        if not np.array_equal(mobile_masks[i], whole_chain):
          raise ValueError(
            'Rigid mobility mask must contain the complete endpoint chain.'
          )
    # Opposite endpoint order describes the same physical distance.
    key = tuple(
      sorted((int(t), tuple(np.flatnonzero(m))) for t, m in zip(tokens, masks))
    )
    previous = intersections.get(key, (0.0, np.inf))
    interval = max(previous[0], lower[i]), min(previous[1], upper[i])
    if interval[0] > interval[1]:
      raise ValueError('Conflicting bounds for the same contact endpoints.')
    intersections[key] = interval


def validate_inference_result(result, batch=None, config=None):
  """Reject nonfinite coordinates and unmet required contacts, at full precision.

  A zero-guidance run is an explicit negative control: it is still checked for
  finite coordinates but is not required to satisfy the supplied contacts.
  """
  samples = result['diffusion_samples']
  counts = samples.get('contact_guidance_nonfinite_counts')
  if counts is not None:
    locations = np.argwhere(np.asarray(counts) > 0)  # step, sample, stage
    if len(locations):
      step, sample, stage = locations[0]
      names = (
        'noisy input',
        'denoiser before guidance',
        'after guidance',
        'Euler output',
      )
      raise FloatingPointError(
        f'Nonfinite AF3 coordinates at step={step}, sample={sample}, stage={names[stage]}.'
      )
  positions = np.asarray(samples['atom_positions'])
  valid = np.asarray(samples['mask'], bool)[..., None]
  if np.any(~np.isfinite(positions) & valid):
    raise FloatingPointError('Nonfinite final valid atom coordinates.')
  weight = (
    config.guidance_weight
    if config is not None
    else result.get('contact_guidance_weight', 0.0)
  )
  if batch is None or weight == 0:
    return
  active = np.asarray(batch.get('contact_constraint_mask', []), bool)
  if not active.any():
    return
  distances = np.asarray(result['contact_restraint_distances'])[:, active]
  lower = np.asarray(
    batch.get('contact_constraint_min_distances', np.zeros(active.shape))
  )[active]
  upper = np.asarray(batch['contact_constraint_max_distances'])[active]
  failures = np.argwhere(
    ~np.isfinite(distances) | (distances < lower) | (distances > upper)
  )
  if len(failures):
    indices = np.flatnonzero(active)
    details = '; '.join(
      f'sample={sample}, contact={indices[row]}, distance={distances[sample, row]:.6g} A, '
      f'bounds=[{lower[row]:.6g}, {upper[row]:.6g}] A'
      for sample, row in failures
    )
    raise ValueError(f'Required contact restraints unsatisfied: {details}')


def restraint_provenance(batch, sample, result):
  """Full-precision, ordered endpoint definitions and production settings."""
  from alphafold3.model import feat_batch

  parsed = feat_batch.Batch.from_data_dict(batch)
  constraints = parsed.contact_constraints
  layout = parsed.convert_model_output.token_atoms_layout
  active = np.asarray(constraints.mask, bool)
  if not active.any():
    return {}
  definitions = []
  for row in np.flatnonzero(active):
    endpoints = []
    for side in (0, 1):
      token = int(constraints.token_indices[row, side])
      selected = np.asarray(constraints.atom_masks[row, side], bool)
      first = int(np.flatnonzero(selected)[0])
      endpoints.append(
        {
          'chain_id': str(layout.chain_id[token, first]),
          'residue_id': int(layout.res_id[token, first]),
          'atom_names': np.asarray(layout.atom_name[token])[selected].tolist(),
        }
      )
    definitions.append(
      {
        'endpoints': endpoints,
        'min_distance': float(constraints.min_distances[row]),
        'max_distance': float(constraints.max_distances[row]),
        'weight': float(constraints.weights[row]),
        'mobile_endpoint': ('both', 'token1', 'token2')[
          int(constraints.mobile_endpoints[row])
        ],
      }
    )
  distances = np.asarray(result['contact_restraint_distances'])[sample, active]
  return {
    'algorithm_baseline': 'ef5dc91d65b4845f63e95ea03316f1613c6dfb5e',
    'constraints': definitions,
    'settings': result.get('contact_guidance_config', {}),
    'distances_A': distances.tolist(),
    'satisfied': (
      (distances >= constraints.min_distances[active])
      & (distances <= constraints.max_distances[active])
    ).tolist(),
    'traces': {
      name: np.asarray(result['diffusion_samples'][name])[:, sample, active].tolist()
      for name in (
        'contact_restraint_distances_before_guidance',
        'contact_restraint_distances_after_guidance',
        'contact_restraint_distances_after_step',
      )
    },
  }
