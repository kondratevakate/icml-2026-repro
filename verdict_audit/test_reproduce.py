import reproduce as R


def test_diff_paths_reports_changed_and_missing_keys():
    old = {"a": 1, "b": {"c": [1, 2], "d": "x"}, "gone": 0}
    new = {"a": 1, "b": {"c": [1, 3], "d": "x"}, "added": 0}
    d = R.diff_paths(new, old)
    assert any(p.startswith(".b.c[1]") for p in d)
    assert ".added (only in new)" in d and ".gone (only in committed)" in d
    assert R.diff_paths(old, old) == []


def test_steps_name_existing_scripts():
    for name, folder, commands, compared, rewritten, ignore in R.STEPS:
        for c in commands:
            assert (R.HERE / folder / c[0]).exists(), (name, c[0])


def test_run_step_restores_a_fresh_output_folder(tmp_path):
    script = tmp_path / "once.py"
    script.write_text("import json, pathlib\nout = pathlib.Path('results')\nout.mkdir(exist_ok=False)\n"
                      "(out / 'r.json').write_text(json.dumps({'x': 1}))\n", encoding="utf-8")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text('{"x": 1}', encoding="utf-8")
    (tmp_path / "results" / "other.json").write_text("{}", encoding="utf-8")
    rec = R.run_step("t", str(tmp_path), [["once.py"]], ["results/r.json"], ["results/"], {}, tmp_path / "logs")
    assert rec["status"] == "match", rec
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["other.json", "r.json"]
    script.write_text(script.read_text(encoding="utf-8").replace("'x': 1", "'x': 2"), encoding="utf-8")
    rec = R.run_step("t", str(tmp_path), [["once.py"]], ["results/r.json"], ["results/"], {}, tmp_path / "logs")
    assert rec["status"] == "differs" and (tmp_path / "results" / "r.json").read_text(encoding="utf-8") == '{"x": 1}'


def test_documented_keys_are_left_out_but_reported(tmp_path):
    script = tmp_path / "w.py"
    script.write_text("import json\nopen('r.json', 'w').write(json.dumps({'x': 1}))\n", encoding="utf-8")
    (tmp_path / "r.json").write_text('{"x": 1, "provenance": "by hand"}', encoding="utf-8")
    rec = R.run_step("t", str(tmp_path), [["w.py"]], ["r.json"], [], {".provenance": "added by hand"}, tmp_path / "logs")
    assert rec["status"] == "match" and rec["ignored"]["r.json"]["differences"] == [".provenance (only in committed)"]
