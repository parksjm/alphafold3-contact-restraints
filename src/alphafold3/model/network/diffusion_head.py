# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion Head."""

from collections.abc import Callable
import math

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from alphafold3.model.network import atom_cross_attention
from alphafold3.model.network import diffusion_transformer
from alphafold3.model.network import featurization
from alphafold3.model.network import noise_level_embeddings
import chex
import haiku as hk
import jax
import jax.numpy as jnp


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


def random_rotation(key):
  # Create a random rotation (Gram-Schmidt orthogonalization of two
  # random normal vectors)
  v0, v1 = jax.random.normal(key, shape=(2, 3))
  e0 = v0 / jnp.maximum(1e-10, jnp.linalg.norm(v0))
  v1 = v1 - e0 * jnp.dot(v1, e0, precision=jax.lax.Precision.HIGHEST)
  e1 = v1 / jnp.maximum(1e-10, jnp.linalg.norm(v1))
  e2 = jnp.cross(e0, e1)
  return jnp.stack([e0, e1, e2])


def random_augmentation(
    rng_key: jnp.ndarray,
    positions: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Apply random rigid augmentation.

  Args:
    rng_key: random key
    positions: atom positions of shape (<common_axes>, 3)
    mask: per-atom mask of shape (<common_axes>,)

  Returns:
    Transformed positions with the same shape as input positions.
  """
  rotation_key, translation_key = jax.random.split(rng_key)

  center = utils.mask_mean(
      mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )
  rot = random_rotation(rotation_key)
  translation = jax.random.normal(translation_key, shape=(3,))

  augmented_positions = (
      jnp.einsum(
          '...i,ij->...j',
          positions - center,
          rot,
          precision=jax.lax.Precision.HIGHEST,
      )
      + translation
  )
  return augmented_positions * mask[..., None]


def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
  return (
      SIGMA_DATA
      * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
  )


class ConditioningConfig(base_config.BaseConfig):
  pair_channel: int
  seq_channel: int
  prob: float


class ContactGuidanceConfig(base_config.BaseConfig):
  """Inference-time flat-bottom contact-restraint guidance."""

  guidance_weight: float = 0.5
  final_weight_scale: float = 1.0
  guidance_interval: int = 1
  num_gradient_steps: int = 20

  def __post_init__(self):
    for name in ('guidance_weight', 'final_weight_scale'):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and non-negative.')
    for name in ('guidance_interval', 'num_gradient_steps'):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


class SampleConfig(base_config.BaseConfig):
  steps: int
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  num_samples: int = 1
  contact_guidance: ContactGuidanceConfig = base_config.autocreate()


class DiffusionHead(hk.Module):
  """Denoising Diffusion Head."""

  class Config(
      atom_cross_attention.AtomCrossAttEncoderConfig,
      atom_cross_attention.AtomCrossAttDecoderConfig,
  ):
    """Configuration for DiffusionHead."""

    eval_batch_size: int = 5
    eval_batch_dim_shard_size: int = 5
    conditioning: ConditioningConfig = base_config.autocreate(
        prob=0.8, pair_channel=128, seq_channel=384
    )
    eval: SampleConfig = base_config.autocreate(
        num_samples=5,
        steps=200,
    )
    transformer: diffusion_transformer.Transformer.Config = (
        base_config.autocreate()
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='diffusion_head',
  ):
    self.config = config
    self.global_config = global_config
    super().__init__(name=name)

  @hk.transparent
  def _conditioning(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      noise_level: jnp.ndarray,
      use_conditioning: bool,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    single_embedding = use_conditioning * embeddings['single']
    pair_embedding = use_conditioning * embeddings['pair']

    rel_features = featurization.create_relative_encoding(
        batch.token_features, max_relative_idx=32, max_relative_chain=2
    ).astype(pair_embedding.dtype)
    features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
    pair_cond = hm.Linear(
        self.config.conditioning.pair_channel,
        precision='highest',
        name='pair_cond_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='pair_cond_initial_norm',
        )(features_2d)
    )

    for idx in range(2):
      pair_cond += diffusion_transformer.transition_block(
          pair_cond, 2, self.global_config, name=f'pair_transition_{idx}'
      )

    target_feat = embeddings['target_feat']
    features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
    single_cond = hm.LayerNorm(
        use_fast_variance=False,
        create_offset=False,
        name='single_cond_initial_norm',
    )(features_1d)
    single_cond = hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='single_cond_initial_projection',
    )(single_cond)

    noise_embedding = noise_level_embeddings.noise_embeddings(
        sigma_scaled_noise_level=noise_level / SIGMA_DATA
    )
    single_cond += hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='noise_embedding_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='noise_embedding_initial_norm',
        )(noise_embedding)
    )

    for idx in range(2):
      single_cond += diffusion_transformer.transition_block(
          single_cond, 2, self.global_config, name=f'single_transition_{idx}'
      )

    return single_cond, pair_cond

  def __call__(
      self,
      # positions_noisy.shape: (num_token, max_atoms_per_token, 3)
      positions_noisy: jnp.ndarray,
      noise_level: jnp.ndarray,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      use_conditioning: bool,
  ) -> jnp.ndarray:

    with utils.bfloat16_context():
      # Get conditioning
      trunk_single_cond, trunk_pair_cond = self._conditioning(
          batch=batch,
          embeddings=embeddings,
          noise_level=noise_level,
          use_conditioning=use_conditioning,
      )

      # Extract features
      sequence_mask = batch.token_features.mask
      atom_mask = batch.predicted_structure_info.atom_mask

      # Position features
      act = positions_noisy * atom_mask[..., None]
      act = act / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)

      enc = atom_cross_attention.atom_cross_att_encoder(
          token_atoms_act=act,
          trunk_single_cond=embeddings['single'],
          trunk_pair_cond=trunk_pair_cond,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
      )
      act = enc.token_act

      # Token-token attention
      chex.assert_shape(act, (None, self.config.per_token_channels))
      act = jnp.asarray(act, dtype=jnp.float32)

      act += hm.Linear(
          act.shape[-1],
          precision='highest',
          initializer=self.global_config.final_init,
          name='single_cond_embedding_projection',
      )(
          hm.LayerNorm(
              use_fast_variance=False,
              create_offset=False,
              name='single_cond_embedding_norm',
          )(trunk_single_cond)
      )

      act = jnp.asarray(act, dtype=jnp.float32)
      trunk_single_cond = jnp.asarray(trunk_single_cond, dtype=jnp.float32)
      trunk_pair_cond = jnp.asarray(trunk_pair_cond, dtype=jnp.float32)
      sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)

      transformer = diffusion_transformer.Transformer(
          self.config.transformer, self.global_config
      )
      act = transformer(
          act=act,
          single_cond=trunk_single_cond,
          mask=sequence_mask,
          pair_cond=trunk_pair_cond,
      )
      act = hm.LayerNorm(
          use_fast_variance=False, create_offset=False, name='output_norm'
      )(act)
      # (n_tokens, per_token_channels)

      # (Possibly) atom-granularity decoder
      assert isinstance(enc, atom_cross_attention.AtomCrossAttEncoderOutput)
      position_update = atom_cross_attention.atom_cross_att_decoder(
          token_act=act,
          enc=enc,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
      )

      skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
      out_scaling = (
          noise_level * SIGMA_DATA / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)
      )
    # End `with utils.bfloat16_context()`.

    return (
        skip_scaling * positions_noisy + out_scaling * position_update
    ) * atom_mask[..., None]


def contact_restraint_min_distances(
    positions: jnp.ndarray,
    constraints,
) -> jnp.ndarray:
  """Returns the minimum valid atom-pair distance for each restraint."""
  token_indices = constraints.token_indices
  endpoint1 = positions[token_indices[:, 0]]
  endpoint2 = positions[token_indices[:, 1]]
  displacements = endpoint1[:, :, None, :] - endpoint2[:, None, :, :]
  distances = jnp.sqrt(jnp.sum(jnp.square(displacements), axis=-1) + 1e-8)
  atom_pair_mask = (
      constraints.atom_masks[:, 0, :, None]
      & constraints.atom_masks[:, 1, None, :]
  )
  distances = jnp.where(atom_pair_mask, distances, 1e6)
  return jnp.min(distances, axis=(-2, -1))


def contact_restraint_energy(
    positions: jnp.ndarray,
    constraints,
) -> jnp.ndarray:
  """Computes a flat-bottom linear bounded-distance restraint energy."""
  min_distances = contact_restraint_min_distances(positions, constraints)
  violations = jnp.maximum(
      constraints.min_distances - min_distances, 0.0
  ) + jnp.maximum(min_distances - constraints.max_distances, 0.0)
  active = constraints.mask.astype(positions.dtype)
  return jnp.sum(active * constraints.weights * violations)


def local_contact_restraint_energy(
    positions: jnp.ndarray,
    constraints,
) -> jnp.ndarray:
  """Returns energy for restraints that move endpoint atoms independently."""
  min_distances = contact_restraint_min_distances(positions, constraints)
  violations = jnp.maximum(
      constraints.min_distances - min_distances, 0.0
  ) + jnp.maximum(min_distances - constraints.max_distances, 0.0)
  active = (
      constraints.mask & (constraints.mobile_endpoints == 0)
  ).astype(positions.dtype)
  return jnp.sum(active * constraints.weights * violations)


def _clip_vector_norm(vector: jnp.ndarray, max_norm: float) -> jnp.ndarray:
  """Returns a vector with its Euclidean norm capped at ``max_norm``."""
  norm = jnp.linalg.norm(vector)
  scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-8))
  return vector * scale


def _rotate_about_center(
    positions: jnp.ndarray,
    center: jnp.ndarray,
    rotation_vector: jnp.ndarray,
) -> jnp.ndarray:
  """Applies an exact axis-angle rotation around ``center``."""
  angle = jnp.linalg.norm(rotation_vector)
  axis = rotation_vector / jnp.maximum(angle, 1e-8)
  centered = positions - center
  rotated = (
      centered * jnp.cos(angle)
      + jnp.cross(axis, centered) * jnp.sin(angle)
      + axis
      * jnp.sum(centered * axis, axis=-1, keepdims=True)
      * (1.0 - jnp.cos(angle))
  )
  return rotated + center


def apply_rigid_chain_contact_guidance(
    positions: jnp.ndarray,
    constraints,
    guidance_weight: jnp.ndarray,
) -> jnp.ndarray:
  """Applies capped exact rigid transforms for explicitly mobile chains."""

  def apply_one_constraint(
      constraint_index: int, current_positions: jnp.ndarray
  ) -> jnp.ndarray:
    token_indices = constraints.token_indices[constraint_index]
    endpoint1 = current_positions[token_indices[0]]
    endpoint2 = current_positions[token_indices[1]]
    mask1 = constraints.atom_masks[constraint_index, 0]
    mask2 = constraints.atom_masks[constraint_index, 1]
    coord1 = jnp.sum(endpoint1 * mask1[..., None], axis=0)
    coord2 = jnp.sum(endpoint2 * mask2[..., None], axis=0)
    displacement = coord1 - coord2
    distance = jnp.linalg.norm(displacement)
    below_minimum = distance < constraints.min_distances[constraint_index]
    above_maximum = distance > constraints.max_distances[constraint_index]
    active = (
        constraints.mask[constraint_index]
        & (constraints.mobile_endpoints[constraint_index] > 0)
        & (below_minimum | above_maximum)
    )
    gradient_sign = jnp.where(
        above_maximum, 1.0, jnp.where(below_minimum, -1.0, 0.0)
    )
    gradient1 = (
        gradient_sign
        * constraints.weights[constraint_index]
        * displacement
        / jnp.maximum(distance, 1e-8)
    )
    mobile_endpoint = constraints.mobile_endpoints[constraint_index]
    mobile_gradient = jnp.where(mobile_endpoint == 1, gradient1, -gradient1)
    descent = -mobile_gradient

    group_mask = constraints.mobile_atom_masks[constraint_index]
    group_mask_float = group_mask.astype(current_positions.dtype)
    group_size = jnp.maximum(jnp.sum(group_mask_float), 1.0)
    group_center = (
        jnp.sum(current_positions * group_mask_float[..., None], axis=(0, 1))
        / group_size
    )
    mobile_coord = jnp.where(mobile_endpoint == 1, coord1, coord2)
    lever_arm = mobile_coord - group_center
    translation = _clip_vector_norm(
        0.5 * guidance_weight * descent, max_norm=0.25
    )
    angular_descent = (
        0.5
        * guidance_weight
        * jnp.cross(lever_arm, descent)
        / jnp.maximum(jnp.sum(jnp.square(lever_arm)), 1.0)
    )
    angular_descent = _clip_vector_norm(
        angular_descent, max_norm=jnp.deg2rad(2.0)
    )
    transformed = _rotate_about_center(
        current_positions, group_center, angular_descent
    ) + translation
    return jnp.where(
        (active & group_mask)[..., None], transformed, current_positions
    )

  return jax.lax.fori_loop(
      0, constraints.mask.shape[0], apply_one_constraint, positions
  )


def apply_contact_guidance(
    positions: jnp.ndarray,
    constraints,
    *,
    guidance_weight: jnp.ndarray,
    guidance_interval: int,
    num_gradient_steps: int,
) -> jnp.ndarray:
  """Runs gradient descent on the contact potential for an x0 prediction."""
  # Legacy unrestrained feature dictionaries have zero-sized restraint arrays.
  # Do not trace an indexing loop body for them, even under an inactive cond.
  if constraints.mask.shape[0] == 0:
    return positions

  def guidance_step(step_index, guided_positions):
    # Interval gates whole updates, not just gradient recomputation. The
    # validated AF3 defaults (interval=1, steps=20) perform 20 updates.
    return jax.lax.cond(
        step_index % guidance_interval == 0,
        lambda x: apply_rigid_chain_contact_guidance(
            x
            - guidance_weight
            * jax.grad(local_contact_restraint_energy)(x, constraints),
            constraints,
            guidance_weight,
        ),
        lambda x: x,
        guided_positions,
    )

  return jax.lax.fori_loop(
      0, num_gradient_steps, guidance_step, positions
  )


def contact_guidance_weight(
    step_index: jnp.ndarray,
    num_steps: int,
    config: ContactGuidanceConfig,
) -> jnp.ndarray:
  """Returns the contact-guidance weight for one diffusion step."""
  steering_t = 1.0 - step_index / num_steps
  return jnp.where(
      steering_t > 0.75,
      config.guidance_weight,
      jnp.where(
          steering_t > 0.25,
          0.5 * config.guidance_weight,
          config.final_weight_scale * config.guidance_weight,
      ),
  )


def sample(
    denoising_step: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch: feat_batch.Batch,
    key: jnp.ndarray,
    config: SampleConfig,
) -> dict[str, jnp.ndarray]:
  """Sample using denoiser on batch.

  Args:
    denoising_step: the denoising function.
    batch: the batch
    key: random key
    config: config for the sampling process (e.g. number of denoising steps,
      etc.)

  Returns:
    a dict
      {
         'atom_positions': jnp.array(...)       # shape (<common_axes>, 3)
         'mask': jnp.array(...)                 # shape (<common_axes>,)
      }
    where the <common_axes> are
    (num_samples, num_tokens, max_atoms_per_token)
  """

  mask = batch.predicted_structure_info.atom_mask

  def nonfinite_count(x):
    return jnp.sum((~jnp.isfinite(x)) & mask[..., None])

  def apply_denoising_step(carry, scan_input):
    noise_level, step_index = scan_input
    key, positions, noise_level_prev = carry
    key, key_noise, key_aug = jax.random.split(key, 3)

    positions = random_augmentation(
        rng_key=key_aug, positions=positions, mask=mask
    )

    gamma = config.gamma_0 * (noise_level > config.gamma_min)
    t_hat = noise_level_prev * (1 + gamma)

    noise_scale = config.noise_scale * jnp.sqrt(t_hat**2 - noise_level_prev**2)
    noise = noise_scale * jax.random.normal(key_noise, positions.shape)
    positions_noisy = positions + noise

    positions_denoised = denoising_step(positions_noisy, t_hat)
    before_guidance_nonfinite = nonfinite_count(positions_denoised)

    guidance_config = config.contact_guidance
    distances_before_guidance = contact_restraint_min_distances(
        positions_denoised, batch.contact_constraints
    )
    scheduled_weight = contact_guidance_weight(
        step_index, config.steps, guidance_config
    )
    apply_guidance = (
        (scheduled_weight > 0)
        & jnp.any(batch.contact_constraints.mask)
    )
    positions_denoised = jax.lax.cond(
        apply_guidance,
        lambda x: apply_contact_guidance(
            x,
            batch.contact_constraints,
            guidance_weight=scheduled_weight,
            guidance_interval=guidance_config.guidance_interval,
            num_gradient_steps=guidance_config.num_gradient_steps,
        ),
        lambda x: x,
        positions_denoised,
    )
    distances_after_guidance = contact_restraint_min_distances(
        positions_denoised, batch.contact_constraints
    )
    grad = (positions_noisy - positions_denoised) / t_hat

    d_t = noise_level - t_hat
    positions_out = positions_noisy + config.step_scale * d_t * grad
    distances_after_step = contact_restraint_min_distances(
        positions_out, batch.contact_constraints
    )

    diagnostics = (
        distances_before_guidance,
        distances_after_guidance,
        distances_after_step,
        scheduled_weight,
        jnp.stack([
            nonfinite_count(positions_noisy), before_guidance_nonfinite,
            nonfinite_count(positions_denoised), nonfinite_count(positions_out),
        ]),
    )
    return (key, positions_out, noise_level), diagnostics

  num_samples = config.num_samples

  noise_levels = noise_schedule(jnp.linspace(0, 1, config.steps + 1))

  key, noise_key = jax.random.split(key)
  positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
  positions *= noise_levels[0]

  init = (
      jax.random.split(key, num_samples),
      positions,
      jnp.tile(noise_levels[None, 0], (num_samples,)),
  )

  apply_denoising_step = hk.vmap(
      apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init())
  )
  scan_inputs = (noise_levels[1:], jnp.arange(config.steps))
  result, diagnostics = hk.scan(
      apply_denoising_step, init, scan_inputs, unroll=4
  )
  _, positions_out, _ = result
  (
      distances_before_guidance,
      distances_after_guidance,
      distances_after_step,
      guidance_weights,
      nonfinite_counts,
  ) = diagnostics

  final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))

  return {
      'atom_positions': positions_out,
      'mask': final_dense_atom_mask,
      'contact_restraint_distances_before_guidance': (
          distances_before_guidance
      ),
      'contact_restraint_distances_after_guidance': distances_after_guidance,
      'contact_restraint_distances_after_step': distances_after_step,
      'contact_guidance_weights': guidance_weights,
      'contact_guidance_nonfinite_counts': nonfinite_counts,
  }
