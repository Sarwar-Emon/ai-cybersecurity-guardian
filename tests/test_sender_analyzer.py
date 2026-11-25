import pytest

from src.sender_analyzer import analyze_sender


@pytest.mark.parametrize(
    "email, expected_finding",
    [
        (
            "From: PayPal Support <support@paypa1.com>",
            "Sender domain resembles paypal.com but does not match it."
        ),
        (
            "From: PayPal Security <security@paypal.com.attacker.com>",
            "Sender domain embeds paypal.com inside a different hostname."
        ),
        (
            "From: PayPal Support <support@paypal-support.com>",
            "Sender hostname combines the brand 'paypal' with a security-related keyword."
        ),
        (
            "From: PayPal <support@paypal.com>",
            None
        ),
        (
            "From: PayPal Support <support@secure.paypal.com>",
            None
        ),
    ],
)
def test_sender_domain_detection(email, expected_finding):
    result = analyze_sender(email)

    if expected_finding is None:
        assert not result["findings"]
    else:
        assert expected_finding in result["findings"]


def test_missing_sender_information():
    result = analyze_sender("Subject: Meeting tomorrow")

    assert result["sender"] == ""
    assert result["score"] == 0


def test_display_name_impersonation():
    result = analyze_sender(
        "From: PayPal Support <support@example.com>"
    )

    assert any(
        "Display name mentions paypal" in finding
        for finding in result["findings"]
    )
