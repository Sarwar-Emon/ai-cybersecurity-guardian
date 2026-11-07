from src.email_detector import analyze_email


def test_account_has_been_suspended():
    result = analyze_email("Your account has been suspended.")
    assert "Account suspension or lockout threat detected." in result["findings"]


def test_account_will_be_suspended():
    result = analyze_email("Your account will be suspended.")
    assert "Account suspension or lockout threat detected." in result["findings"]


def test_unrelated_message_not_flagged():
    result = analyze_email("Your meeting is scheduled for tomorrow.")
    assert "Account suspension or lockout threat detected." not in result["findings"]
