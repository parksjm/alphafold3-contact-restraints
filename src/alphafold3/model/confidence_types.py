# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Confidence categories for predictions."""

import dataclasses
import enum
import json
from typing import Any, Self

from absl import logging
from alphafold3.model import model
import jax
import numpy as np


class StructureConfidenceFullEncoder(json.JSONEncoder):
  """JSON encoder for serializing confidence types."""

  def __init__(self, **kwargs):
    super().__init__(**(kwargs | dict(separators=(',', ':'))))

  def encode(self, o: 'StructureConfidenceFull'):
    # Cast to np.float64 before rounding, since casting to Python float will
    # cast to a 64 bit float, potentially undoing np.float32 rounding.
    atom_plddts = np.round(
        np.clip(np.asarray(o.atom_plddts, dtype=np.float64), 0.0, 99.99), 2
    ).astype(float)
    contact_probs = np.round(
        np.clip(np.asarray(o.contact_probs, dtype=np.float64), 0.0, 1.0), 2
    ).astype(float)
    pae = np.round(
        np.clip(np.asarray(o.pae, dtype=np.float64), 0.0, 99.9), 1
    ).astype(float)
    contact_restraint_distances = np.round(
        np.asarray(o.contact_restraint_distances, dtype=np.float64), 2
    ).astype(float)
    contact_restraint_thresholds = np.round(
        np.asarray(o.contact_restraint_thresholds, dtype=np.float64), 2
    ).astype(float)
    contact_restraint_min_thresholds = np.round(
        np.asarray(o.contact_restraint_min_thresholds, dtype=np.float64), 2
    ).astype(float)
    contact_restraint_distances_before_guidance = np.round(
        np.asarray(
            o.contact_restraint_distances_before_guidance, dtype=np.float64
        ),
        2,
    ).astype(float)
    contact_restraint_distances_after_guidance = np.round(
        np.asarray(
            o.contact_restraint_distances_after_guidance, dtype=np.float64
        ),
        2,
    ).astype(float)
    contact_restraint_distances_after_step = np.round(
        np.asarray(o.contact_restraint_distances_after_step, dtype=np.float64),
        2,
    ).astype(float)
    contact_guidance_weights = np.round(
        np.asarray(o.contact_guidance_weights, dtype=np.float64), 4
    ).astype(float)
    return """\
{
  "atom_chain_ids": %s,
  "atom_plddts": %s,
  "contact_guidance_weights": %s,
  "contact_probs": %s,
  "contact_restraint_distances": %s,
  "contact_restraint_distances_after_guidance": %s,
  "contact_restraint_distances_after_step": %s,
  "contact_restraint_distances_before_guidance": %s,
  "contact_restraint_min_thresholds": %s,
  "contact_restraint_satisfied": %s,
  "contact_restraint_thresholds": %s,
  "pae": %s,
  "token_chain_ids": %s,
  "token_res_ids": %s
}""" % (
        super().encode(o.atom_chain_ids),
        super().encode(list(atom_plddts)).replace('NaN', 'null'),
        super().encode(contact_guidance_weights.tolist()),
        super().encode([list(x) for x in contact_probs]).replace('NaN', 'null'),
        super().encode(list(contact_restraint_distances)),
        super().encode(contact_restraint_distances_after_guidance.tolist()),
        super().encode(contact_restraint_distances_after_step.tolist()),
        super().encode(contact_restraint_distances_before_guidance.tolist()),
        super().encode(list(contact_restraint_min_thresholds)),
        super().encode(o.contact_restraint_satisfied),
        super().encode(list(contact_restraint_thresholds)),
        super().encode([list(x) for x in pae]).replace('NaN', 'null'),
        super().encode(o.token_chain_ids),
        super().encode(o.token_res_ids),
    )


def _dump_json(data: Any, indent: int | None = None) -> str:
  """Dumps a json string with JSON compatible NaN representation."""
  json_str = json.dumps(
      data,
      sort_keys=True,
      indent=indent,
      separators=(',', ': '),
  )
  return json_str.replace('NaN', 'null')


@enum.unique
class ConfidenceCategory(enum.Enum):
  """Confidence categories for AlphaFold predictions."""

  HIGH = 0
  MEDIUM = 1
  LOW = 2
  DISORDERED = 3

  @classmethod
  def from_char(cls, char: str) -> Self:
    match char:
      case 'H':
        return cls.HIGH
      case 'M':
        return cls.MEDIUM
      case 'L':
        return cls.LOW
      case 'D':
        return cls.DISORDERED
      case _:
        raise ValueError(
            f'Unknown character. Expected one of H, M, L or D; got: {char}'
        )

  def to_char(self) -> str:
    match self:
      case self.HIGH:
        return 'H'
      case self.MEDIUM:
        return 'M'
      case self.LOW:
        return 'L'
      case self.DISORDERED:
        return 'D'

  @classmethod
  def from_confidence_score(cls, confidence: float) -> Self:
    if 90 <= confidence <= 100:
      return cls.HIGH
    if 70 <= confidence < 90:
      return cls.MEDIUM
    if 50 <= confidence < 70:
      return cls.LOW
    if 0 <= confidence < 50:
      return cls.DISORDERED
    raise ValueError(f'Confidence score out of range [0, 100]: {confidence}')


@dataclasses.dataclass()
class AtomConfidence:
  """Dataclass for 1D per-atom confidences from AlphaFold."""

  chain_id: list[str]
  atom_number: list[int]
  confidence: list[float]
  confidence_category: list[ConfidenceCategory]

  def __post_init__(self):
    num_res = len(self.atom_number)
    if not all(
        len(v) == num_res
        for v in [self.chain_id, self.confidence, self.confidence_category]
    ):
      raise ValueError('All confidence fields must have the same length.')

  @classmethod
  def from_inference_result(
      cls, inference_result: model.InferenceResult
  ) -> Self:
    """Instantiates an AtomConfidence from a structure.

    Args:
      inference_result: Inference result from AlphaFold.

    Returns:
      Scores in AtomConfidence dataclass.
    """
    struc = inference_result.predicted_structure
    as_dict = {
        'chain_id': [],
        'atom_number': [],
        'confidence': [],
        'confidence_category': [],
    }
    for atom_number, atom in enumerate(struc.iter_atoms()):
      this_confidence = float(struc.atom_b_factor[atom_number])
      as_dict['chain_id'].append(atom['chain_id'])
      as_dict['atom_number'].append(atom_number)
      as_dict['confidence'].append(round(this_confidence, 2))
      as_dict['confidence_category'].append(
          ConfidenceCategory.from_confidence_score(this_confidence)
      )
    return cls(**as_dict)

  @classmethod
  def from_json(cls, json_string: str) -> Self:
    """Instantiates a AtomConfidence from a json string."""
    input_dict = json.loads(json_string)
    input_dict['confidence_category'] = [
        ConfidenceCategory.from_char(k)
        for k in input_dict['confidence_category']
    ]
    return cls(**input_dict)

  def to_json(self) -> str:
    output = dataclasses.asdict(self)
    output['confidence_category'] = [
        k.to_char() for k in output['confidence_category']
    ]
    output['atom_number'] = [int(k) for k in output['atom_number']]
    return _dump_json(output)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StructureConfidenceSummary:
  """Dataclass for the summary of structure scores from AlphaFold.

  Attributes:
   ptm: Predicted TM global score.
   iptm: Interface predicted TM global score.
   ranking_score: Ranking score extracted from CIF metadata.
   fraction_disordered: Fraction disordered, measured with RASA.
   has_clash: Has significant clashing.
   chain_pair_pae_min: [num_chains, num_chains] Minimum cross chain PAE.
   chain_pair_iptm: [num_chains, num_chains] Chain pair ipTM.
   chain_ptm: [num_chains] Chain pTM.
   chain_iptm: [num_chains] Mean cross chain ipTM for a chain.
  """

  ptm: float
  iptm: float
  ranking_score: float
  fraction_disordered: float
  has_clash: float
  chain_pair_pae_min: np.ndarray
  chain_pair_iptm: np.ndarray
  chain_ptm: np.ndarray
  chain_iptm: np.ndarray
  actifptm: float
  chain_pair_actifptm: np.ndarray
  chain_actifptm: np.ndarray
  

  @classmethod
  def from_inference_result(
      cls, inference_result: model.InferenceResult
  ) -> Self:
    """Returns a new instance based on a given inference result."""
    return cls(
        ptm=float(inference_result.metadata['ptm']),
        iptm=float(inference_result.metadata['iptm']),
        ranking_score=float(inference_result.metadata['ranking_score']),
        fraction_disordered=float(
            inference_result.metadata['fraction_disordered']
        ),
        has_clash=float(inference_result.metadata['has_clash']),
        chain_pair_pae_min=inference_result.metadata['chain_pair_pae_min'],
        chain_pair_iptm=inference_result.metadata['chain_pair_iptm'],
        chain_ptm=inference_result.metadata['iptm_ichain'],
        chain_iptm=inference_result.metadata['iptm_xchain'],
	actifptm=float(inference_result.metadata['actifptm']),
	chain_pair_actifptm=inference_result.metadata['chain_pair_actifptm'],
	chain_actifptm=inference_result.metadata['actifptm_xchain'],
    )

  @classmethod
  def from_json(cls, json_string: str) -> Self:
    """Returns a new instance from a given json string."""
    return cls(**json.loads(json_string))

  def to_json(self) -> str:
    def convert(data):
      if isinstance(data, np.ndarray):
        # Cast to np.float64 before rounding, since casting to Python float will
        # cast to a 64 bit float, potentially undoing np.float32 rounding.
        rounded_data = np.round(data.astype(np.float64), decimals=2).tolist()
      else:
        rounded_data = np.round(data, decimals=2)
      return rounded_data

    return _dump_json(jax.tree.map(convert, dataclasses.asdict(self)), indent=1)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StructureConfidenceFull:
  """Dataclass for full structure data from AlphaFold."""

  pae: np.ndarray
  token_chain_ids: list[str]
  token_res_ids: list[int]
  atom_plddts: list[float]
  atom_chain_ids: list[str]
  contact_probs: np.ndarray  # [num_tokens, num_tokens]
  contact_restraint_distances: list[float] = dataclasses.field(
      default_factory=list
  )
  contact_restraint_thresholds: list[float] = dataclasses.field(
      default_factory=list
  )
  contact_restraint_min_thresholds: list[float] = dataclasses.field(
      default_factory=list
  )
  contact_restraint_satisfied: list[bool] = dataclasses.field(
      default_factory=list
  )
  contact_restraint_distances_before_guidance: list[list[float]] = (
      dataclasses.field(default_factory=list)
  )
  contact_restraint_distances_after_guidance: list[list[float]] = (
      dataclasses.field(default_factory=list)
  )
  contact_restraint_distances_after_step: list[list[float]] = dataclasses.field(
      default_factory=list
  )
  contact_guidance_weights: list[float] = dataclasses.field(
      default_factory=list
  )

  @classmethod
  def from_inference_result(
      cls, inference_result: model.InferenceResult
  ) -> Self:
    """Returns a new instance based on a given inference result."""

    pae = inference_result.numerical_data['full_pae']
    if not isinstance(pae, np.ndarray):
      logging.info('%s', type(pae))
      raise TypeError('pae should be a numpy array.')

    contact_probs = inference_result.numerical_data['contact_probs']
    if not isinstance(contact_probs, np.ndarray):
      logging.info('%s', type(contact_probs))
      raise TypeError('contact_probs should be a numpy array.')

    contact_restraint_distances = np.asarray(
        inference_result.numerical_data.get('contact_restraint_distances', [])
    ).tolist()
    contact_restraint_thresholds = np.asarray(
        inference_result.numerical_data.get('contact_restraint_thresholds', [])
    ).tolist()
    contact_restraint_min_thresholds = np.asarray(
        inference_result.numerical_data.get(
            'contact_restraint_min_thresholds', []
        )
    ).tolist()
    contact_restraint_satisfied = np.asarray(
        inference_result.numerical_data.get('contact_restraint_satisfied', []),
        dtype=bool,
    ).tolist()
    contact_restraint_distances_before_guidance = np.asarray(
        inference_result.numerical_data.get(
            'contact_restraint_distances_before_guidance', []
        )
    ).tolist()
    contact_restraint_distances_after_guidance = np.asarray(
        inference_result.numerical_data.get(
            'contact_restraint_distances_after_guidance', []
        )
    ).tolist()
    contact_restraint_distances_after_step = np.asarray(
        inference_result.numerical_data.get(
            'contact_restraint_distances_after_step', []
        )
    ).tolist()
    contact_guidance_weights = np.asarray(
        inference_result.numerical_data.get('contact_guidance_weights', [])
    ).tolist()

    struc = inference_result.predicted_structure
    chain_ids = struc.chain_id.tolist()
    atom_plddts = struc.atom_b_factor.tolist()
    token_chain_ids = [
        str(token_id)
        for token_id in inference_result.metadata['token_chain_ids']
    ]
    token_res_ids = [
        int(token_id) for token_id in inference_result.metadata['token_res_ids']
    ]
    return cls(
        pae=pae,
        token_chain_ids=token_chain_ids,
        token_res_ids=token_res_ids,
        atom_plddts=atom_plddts,
        atom_chain_ids=chain_ids,
        contact_probs=contact_probs,
        contact_restraint_distances=contact_restraint_distances,
        contact_restraint_thresholds=contact_restraint_thresholds,
        contact_restraint_min_thresholds=contact_restraint_min_thresholds,
        contact_restraint_satisfied=contact_restraint_satisfied,
        contact_restraint_distances_before_guidance=(
            contact_restraint_distances_before_guidance
        ),
        contact_restraint_distances_after_guidance=(
            contact_restraint_distances_after_guidance
        ),
        contact_restraint_distances_after_step=(
            contact_restraint_distances_after_step
        ),
        contact_guidance_weights=contact_guidance_weights,
    )

  @classmethod
  def from_json(cls, json_string: str) -> Self:
    """Returns a new instance from a given json string."""
    return cls(**json.loads(json_string))

  def to_json(self) -> str:
    """Converts StructureConfidenceFull to json string."""
    return json.dumps(self, cls=StructureConfidenceFullEncoder)
