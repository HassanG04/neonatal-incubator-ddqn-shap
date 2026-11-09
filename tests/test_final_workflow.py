"""Regression checks for formerly sparse records, hidden errors and mixed methods."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest

import run_member03
import shap_explainer as xai
from environment import NeonatalIncubatorEnv
from rl_utils import AgentPolicy
from validation import evaluate_trajectory

CHECKPOINT = str(Path(__file__).resolve().parents[1] / "models/ddqn_tuned_seed0/best.pt")


@pytest.fixture(scope="module")
def model_background():
    agent = xai.load_trained_agent(CHECKPOINT)
    _, background, _ = xai.collect_background_dataset(agent=agent)
    return agent, background


def test_deep_failure_during_preflight_uses_full_kernel_background(model_background):
    agent, background = model_background
    with patch.object(xai.shap.DeepExplainer, "shap_values", side_effect=RuntimeError("preflight probe")):
        explainer = xai.DDQNShapExplainer(agent, background)
    assert explainer.explainer_used == "kernel"
    assert "preflight probe" in explainer.fallback_reason
    np.testing.assert_allclose(explainer.expected_values, agent.q_values(background).mean(axis=0), atol=1e-5)
    x = background[[12, 55]]
    values = explainer.explain_norm_batch(x)
    assert values.shape == (2, 11, 25)
    np.testing.assert_allclose(values.sum(axis=1) + explainer.expected_values, agent.q_values(x), atol=1e-4)
    np.testing.assert_array_equal(values[0], explainer.explain_norm_batch(x[:1])[0])


def test_failures_during_validation_propagate(model_background):
    agent, _ = model_background
    class Broken:
        def decision_values(self, *args):
            raise RuntimeError("decision-time failure")
    with pytest.raises(RuntimeError, match="decision-time failure"):
        evaluate_trajectory(NeonatalIncubatorEnv(max_steps=2), AgentPolicy(agent), 42, Broken())


def test_skip_never_initializes_shap(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("SHAP code executed during validation-only workflow")
    monkeypatch.setattr(xai.DDQNShapExplainer, "__init__", unexpected)
    monkeypatch.setattr(run_member03, "run_shap_analysis", unexpected)
    with TemporaryDirectory(dir=Path.cwd()) as directory:
        tmp_path = Path(directory)
        out = run_member03.run_member03_workflow(CHECKPOINT, directory, 1, skip_shap=True)
        assert out["validation_summary"]["evaluation_design"]["total_evaluation_episodes"] == 10
        assert not (tmp_path / "shap").exists()
        assert "shap_base_val" not in (tmp_path / "validation/normal_scenario_results.csv").read_text().splitlines()[0]

