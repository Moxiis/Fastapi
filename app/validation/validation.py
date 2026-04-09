from typing import Any, Dict, Tuple
from pydantic import ValidationError

from ..api.schemas import HousePriceInput
from ..core.storage import store_raw_input


def validate_and_store(raw_data: Dict[str, Any]) -> Tuple[HousePriceInput, str]:
	"""Validate raw user input using the `HousePriceInput` schema and store
	both the raw and the validated payloads to disk for future training.

	Returns the validated `HousePriceInput` instance and the path where the
	raw/validated data were stored (best-effort).
	"""
	# Best-effort: store the raw payload first
	try:
		store_raw_input({"raw": raw_data})
	except Exception:
		# Do not fail the request if persistence is temporarily unavailable
		pass

	# Validate with Pydantic schema (this will raise ValidationError on failure)
	validated = HousePriceInput(**raw_data)

	# Store the validated (cleaned) payload too
	try:
		store_raw_input({"validated": validated.model_dump()})
	except Exception:
		pass

	return validated, "stored"
