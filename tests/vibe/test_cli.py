import json

from dspy_vibe.cli import main

INSTRUCTION = "csinálj egy login formot, ne legyen backend"


def test_convert_writes_all_three_artifacts(tmp_path, capsys):
    code = main(["convert", INSTRUCTION, "--out", str(tmp_path)])
    assert code == 0
    written = sorted(path.suffix for path in tmp_path.iterdir())
    assert written == [".agent", ".md", ".skill"]
    assert "bundle quality" in capsys.readouterr().out


def test_convert_is_offline_by_default(tmp_path, monkeypatch):
    # No LM configuration is read and no network client is constructed.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["convert", INSTRUCTION, "--out", str(tmp_path)]) == 0


def test_json_format_produces_machine_readable_artifacts(tmp_path):
    main(["convert", INSTRUCTION, "--out", str(tmp_path), "--format", "json"])
    agent = next(tmp_path.glob("*.agent"))
    assert json.loads(agent.read_text(encoding="utf-8"))["name"].endswith("-agent")


def test_second_run_refuses_to_overwrite(tmp_path, capsys):
    main(["convert", INSTRUCTION, "--out", str(tmp_path)])
    assert main(["convert", INSTRUCTION, "--out", str(tmp_path)]) == 1
    assert "refusing to overwrite" in capsys.readouterr().err
    assert main(["convert", INSTRUCTION, "--out", str(tmp_path), "--overwrite"]) == 0


def test_stdout_mode_prints_the_bundle_without_writing(tmp_path, capsys):
    assert main(["convert", INSTRUCTION, "--out", str(tmp_path), "--stdout"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"spec", "agent", "skill"} <= payload.keys()
    assert not list(tmp_path.iterdir())


def test_instruction_can_come_from_a_file(tmp_path):
    source = tmp_path / "request.txt"
    source.write_text(INSTRUCTION, encoding="utf-8")
    assert main(["convert", "--file", str(source), "--out", str(tmp_path / "out")]) == 0


def test_check_validates_json_artifacts(tmp_path, capsys):
    main(["convert", INSTRUCTION, "--out", str(tmp_path), "--format", "json"])
    agent = next(tmp_path.glob("*.agent"))
    assert main(["check", str(agent)]) == 0
    assert "PASS" in capsys.readouterr().out


def test_check_reports_an_invalid_artifact(tmp_path, capsys):
    broken = tmp_path / "broken.agent"
    broken.write_text(json.dumps({"name": "Not A Slug"}), encoding="utf-8")
    assert main(["check", str(broken)]) == 1
    assert "FAIL" in capsys.readouterr().out
