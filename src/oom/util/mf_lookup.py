from typing import Any, Optional

import scipy as sp

from src.oom.discrete_observable import DiscreteObservable


class _MfLookup:
	def __init__(
		self,
		mfn_dict: dict[DiscreteObservable, sp.stats.rv_continuous],
		steps_pdfs: int = 10000,
		steps_rvs: int = 10000
	):
		self._mfn_dict: dict[DiscreteObservable, sp.stats.rv_continuous] = mfn_dict
		
		self._steps_pdfs = steps_pdfs
		self._pdf_evals = dict.fromkeys(mfn_dict.keys())
		self.seqrange: tuple[int, int] = (0, 0)
		
		self._steps_rvs = steps_rvs
		self._rvs = dict.fromkeys(mfn_dict.keys())
		for obs in self._rvs:
			self._rvs[obs] = {
				"rvs": None,
				"pdfs": dict.fromkeys(mfn_dict.keys())
			}
		self.rvsizes = dict.fromkeys(mfn_dict.keys(), 0)
		self._last_rv: DiscreteObservable = None
	
	def update_pdfs(
		self,
		sequence: float | list[float],
		start: int = 0,
		steps: Optional[int] = None
	) -> None:
		steps = steps if steps is not None else self._steps_pdfs
		nitems = 0
		for obs, mf in self._mfn_dict.items():
			self._pdf_evals[obs] = mf.pdf(sequence[start: start + steps])
			nitems = max(nitems, len(self._pdf_evals[obs]))
		self.seqrange = (start, start + nitems)
	
	def __getitem__(
		self,
		item: tuple[DiscreteObservable, Any]
	):
		obs, pos = item
		return self._pdf_evals[obs][pos - self.seqrange[0]]
	
	def holds_pdfs(self, start: int, end: int) -> bool:
		return start >= self.seqrange[0] and end < self.seqrange[1]
	
	
	def update_rvs(
		self,
		steps: Optional[int] = None,
		obs: Optional[DiscreteObservable] = None
	) -> None:
		"""
		
		"""
		steps = steps if steps is not None else self._steps_pdfs
		targets = {obs: self._mfn_dict[obs]} if obs is not None else self._mfn_dict
		
		for tobs, tmf in targets.items():
			# Precompute RVs
			self._rvs[tobs]["rvs"] = tmf.rvs(steps)
			self.rvsizes[tobs] = steps
			
			# Precompute their PDFs
			for sobs, smf in self._mfn_dict.items():
				self._rvs[tobs]["pdfs"][sobs] = smf.pdf(self._rvs[tobs]["rvs"])
		
		return
	
	def __call__(
		self,
		goal: str,
		obs: Optional[DiscreteObservable] = None
	) -> float | list[float]:
		"""
		
		"""
		match goal:
			case "rvs":
				# Get an RV for the desired observable from the precomputed ones
				self._last_rv = obs
				rv = self._rvs[obs]["rvs"][self.rvsizes[obs] - 1]
				self.rvsizes[obs] -= 1
				return rv
			case "pdfs":
				# Given we know the last sampled RV, get its precomputed PDFs
				return [
					self._rvs[self._last_rv]\
						["pdfs"]\
						[either_obs]\
						[self.rvsizes[self._last_rv]]
					for either_obs in self._mfn_dict.keys()
				]
			case _:
				raise ValueError("'goal' argument must be either 'rvs' or 'pdfs'.")
	
	def holds_rvs(self, obs: DiscreteObservable) -> bool:
		return self.rvsizes[obs] > 0