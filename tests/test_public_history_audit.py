import json

from scripts.audit_public_history import inspect_text


def test_audit_reports_location_without_echoing_candidate():
    candidate = "ghp_" + "A" * 36
    result = inspect_text("first line\nTOKEN = '" + candidate + "'\n")
    assert result[0]["category"] == "github_token"
    assert result[0]["lines"] == [2]
    assert candidate not in json.dumps(result)


def test_audit_detects_literal_secret_but_not_environment_lookup():
    sample = "password" + ' = "' + "example-long-secret" + '"'
    assert any(r["category"] == "assigned_secret" for r in inspect_text(sample))
    assert inspect_text('password = os.environ["SERVICE_PASSWORD"]') == []


def test_audit_flags_full_account_not_short_suffix():
    sample = "acc_id: " + str(123456789)
    assert any(r["category"] == "full_account_id" for r in inspect_text(sample))
    assert inspect_text('account_suffix: "1234"') == []
