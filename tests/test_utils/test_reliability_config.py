import pytest

from tests.reliability.utils import parse_reliability_conf_yaml


def test_reliability_config_uses_native_lm_parameters(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("""adapter: chat
model_list:
  - model_name: judge
    lm_params:
      model: openai/gpt-4o
      temperature: 0
  - model_name: unconfigured
    lm_params:
""")
    config = parse_reliability_conf_yaml(str(path))
    assert config.adapter == "chat"
    assert config.models == {"judge": {"model": "openai/gpt-4o", "temperature": 0}}


def test_reliability_config_reports_invalid_configuration(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("adapter: chat\nmodel_list:\n  - lm_params: {}\n")
    with pytest.raises(ValueError, match="Error parsing LM configuration") as exc:
        parse_reliability_conf_yaml(str(path))
    assert "Model name missing" in str(exc.value.__cause__)
