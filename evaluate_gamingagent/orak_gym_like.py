"""
Backward-compatible shim -- re-exports from evaluate_orak.orak_gym_like.

All Orak Gymnasium wrapper code now lives in evaluate_orak/orak_gym_like.py.
"""
from evaluate_orak.orak_gym_like import (  # noqa: F401
    ORAK_GAME_CONFIG_MAPPING,
    list_orak_games,
    make_orak_gaming_env,
)
