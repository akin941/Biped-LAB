import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "average_checkpoints",
    Path(__file__).resolve().parents[2] / "scripts" / "average_checkpoints.py",
)
avg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avg_mod)


def _ckpt(w: float, var: float, count: int, it: int) -> dict:
    def sd() -> OrderedDict:
        return OrderedDict(
            [
                ("obs_normalizer._mean", torch.full((1, 3), w)),
                ("obs_normalizer._var", torch.full((1, 3), var)),
                ("obs_normalizer._std", torch.full((1, 3), var**0.5)),
                ("obs_normalizer.count", torch.tensor(count)),
                ("mlp.0.weight", torch.full((2, 3), w)),
            ]
        )

    return {
        "actor_state_dict": sd(),
        "critic_state_dict": sd(),
        "optimizer_state_dict": {"it": it},
        "iter": it,
        "infos": {"iterations": it},
    }


def test_average_weights_and_normaliser():
    out = avg_mod.average([_ckpt(1.0, 1.0, 10, 1), _ckpt(3.0, 9.0, 20, 2)])
    a = out["actor_state_dict"]
    assert torch.allclose(a["mlp.0.weight"], torch.full((2, 3), 2.0))
    assert torch.allclose(a["obs_normalizer._var"], torch.full((1, 3), 5.0))
    assert torch.allclose(a["obs_normalizer._std"], torch.full((1, 3), 5.0**0.5))
    assert int(a["obs_normalizer.count"]) == 20
    assert out["infos"] == {"iterations": 2} and out["optimizer_state_dict"] == {"it": 2}
    assert torch.allclose(out["critic_state_dict"]["mlp.0.weight"], torch.full((2, 3), 2.0))


def test_average_rejects_mismatched_checkpoints():
    bad = _ckpt(1.0, 1.0, 1, 1)
    bad["actor_state_dict"]["mlp.0.weight"] = torch.zeros(4, 3)
    with pytest.raises(ValueError):
        avg_mod.average([_ckpt(1.0, 1.0, 1, 1), bad])
    with pytest.raises(ValueError):
        avg_mod.average([])
