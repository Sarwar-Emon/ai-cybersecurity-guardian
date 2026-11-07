import streamlit as st
from src.sms_detector import analyze_sms
from src.email_detector import analyze_email
from src.ml_detector import predict_url
from src.risk_scoring import calculate_risk
from src.url_detector import analyze_url
from src.url_validator import validate_url
from src.ollama_explainer import explain_security_findings

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Cybersecurity Guardian",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ AI Cybersecurity Guardian")
st.write(
    "Analyze suspicious URLs, emails, and messages "
    "using cybersecurity rules and machine learning."
)


# ==========================================
# 2. USER INPUT
# ==========================================
input_type = st.selectbox(
    "Select Input Type",
    ["Email", "Text Message", "URL"],
)

user_input = st.text_area(
    "Enter content to analyze",
    height=180,
    placeholder="Paste suspicious content here...",
)


# ==========================================
# 3. ANALYSIS
# ==========================================

# ==========================================
# 3. ANALYSIS
# ==========================================

if "submitted_analysis" not in st.session_state:
    st.session_state.submitted_analysis = None

if st.button("Analyze Threat"):
    if not user_input.strip():
        st.session_state.submitted_analysis = None
        st.warning("Please enter content first.")
    else:
        st.session_state.submitted_analysis = {
            "input_type": input_type,
            "user_input": user_input,
        }

if st.session_state.submitted_analysis is not None:

    input_type = st.session_state.submitted_analysis["input_type"]
    user_input = st.session_state.submitted_analysis["user_input"]

    if not user_input.strip():
        st.warning("Please enter content first.")


    elif input_type == "URL":
        # ----------------------------------
        # VALIDATION AND NORMALIZATION
        # ----------------------------------
        validation = validate_url(user_input)

        if not validation["valid"]:
            st.error(validation["message"])
            st.stop()

        url = validation["url"]

        if validation["normalized"]:
            st.warning(validation["message"])

        st.caption(f"URL analyzed: {url}")

        # ----------------------------------
        # RULE-BASED ANALYSIS
        # ----------------------------------
        try:
            findings = analyze_url(url)
            risk = calculate_risk(findings)
        except Exception as error:
            st.error("URL analysis could not be completed.")
            st.code(str(error))
            st.stop()

        if risk is None:
            st.error(
                findings[0]
                if findings
                else "Invalid URL. Please enter a valid URL."
            )
        else:
            st.subheader("🛡️ URL Security Analysis")
            col1, col2 = st.columns(2)
            ml_result = None
            rule_flagged = len(findings) > 0

            # ----------------------------------
            # RULE-BASED RESULTS
            # ----------------------------------
            with col1:
                st.markdown("### Rule-Based Detection")
                st.metric("Heuristic Risk Score", f"{risk['score']}/100")

                if risk["level"] == "High Risk":
                    st.error("🔴 High Heuristic Risk")
                elif risk["level"] == "Medium Risk":
                    st.warning("🟠 Medium Heuristic Risk")
                else:
                    st.info("🟢 Low Heuristic Risk")

                if findings:
                    st.markdown("**Potentially Suspicious Indicators:**")
                    for finding in findings:
                        st.write("⚠️", finding)
                else:
                    st.info(
                        "No suspicious indicators detected "
                        "by the current rules."
                    )

                st.caption(
                    "This heuristic score is based on predefined rules, "
                    "not a calibrated phishing probability. Low heuristic "
                    "risk does not necessarily mean zero indicators."
                )

            # ----------------------------------
            # MACHINE LEARNING RESULTS
            # ----------------------------------
            with col2:
                st.markdown("### Machine Learning Detection")
                try:
                    ml_result = predict_url(url)
                    ml_score = ml_result["phishing_score"]

                    st.metric(
                        "ML Phishing-Class Score",
                        f"{ml_score * 100:.2f}%",
                    )

                    if ml_result["prediction"] == "Phishing":
                        st.error("🔴 ML Classification: Phishing")
                    else:
                        st.info("🔵 ML Classification: Legitimate")

                    st.progress(min(max(ml_score, 0.0), 1.0))
                    st.write("**Model:**", ml_result["model"])
                    st.write("**Decision Threshold:**", ml_result["threshold"])
                    st.caption(
                        "The ML score is an uncalibrated classifier output, "
                        "not a verified real-world phishing probability."
                    )
                except Exception as error:
                    ml_result = None
                    st.error("Machine-learning analysis could not be completed.")
                    st.code(str(error))

            # ----------------------------------
            # HYBRID SECURITY ASSESSMENT
            # ----------------------------------
            st.divider()
            st.subheader("🔎 Hybrid Security Assessment")

            if ml_result is None:
                st.warning(
                    "Hybrid assessment is unavailable because "
                    "machine-learning analysis did not complete."
                )
            else:
                ml_flagged = ml_result["prediction"] == "Phishing"

                if rule_flagged and ml_flagged:
                    st.error(
                        "⚠️ Both detectors identified suspicious URL signals."
                    )
                    st.write(
                        "The predefined rules detected potentially suspicious "
                        "indicators, and the ML model classified the URL as phishing."
                    )
                elif ml_flagged and not rule_flagged:
                    st.warning(
                        "⚠️ Machine-learning warning: The ML model classified "
                        "this URL as phishing, although the rules found no "
                        "suspicious indicators."
                    )
                    st.write(
                        "The model may be responding to structural URL patterns "
                        "that are not explicitly covered by the current rules."
                    )
                elif rule_flagged and not ml_flagged:
                    st.warning(
                        "⚠️ Rule-based warning: The security rules detected "
                        "potentially suspicious indicators, although the ML model "
                        "classified this URL as legitimate."
                    )
                    st.write(
                        "The ML classification does not cancel the indicators "
                        "detected by the rules. These indicators are not proof "
                        "of phishing."
                    )
                else:
                    st.info("Neither detector flagged suspicious signals.")
                    st.write(
                        "This does not establish that the URL is safe. Some "
                        "phishing URLs may not be detected by either method."
                    )

            st.caption(
                "The hybrid assessment compares rule indicators with the ML "
                "classification. It does not average their numerical scores "
                "or guarantee correctness."
            )

            # ----------------------------------
            # URL SECURITY RECOMMENDATIONS
            # ----------------------------------
            st.divider()
            st.subheader("🔐 Security Recommendations")

            ml_flagged = (
                ml_result is not None
                and ml_result["prediction"] == "Phishing"
            )

            if rule_flagged or ml_flagged:
                st.warning(
                    "Exercise caution with this URL. At least one detector "
                    "identified potentially suspicious signals."
                )
                st.write(
                    "• Avoid entering passwords, payment information, or "
                    "personal details until the website is verified."
                )
                st.write(
                    "• Verify the website using an independently obtained "
                    "official address."
                )
                st.write("• Do not trust a link solely because it uses HTTPS.")
            elif ml_result is None:
                st.warning(
                    "ML analysis was unavailable. Do not interpret this "
                    "partial assessment as proof of safety."
                )
                st.write(
                    "• Verify the website independently before sharing "
                    "sensitive information."
                )
            else:
                st.info("No suspicious signals were flagged by the available detectors.")
                st.write(
                    "• Verify the website address before entering "
                    "sensitive information."
                )
                st.write(
                    "• Be cautious with unexpected links received "
                    "through email or SMS."
                )

            st.caption(
                "AI Cybersecurity Guardian is an experimental security "
                "analysis tool. Its results do not guarantee that a "
                "website is safe or malicious."
            )


            # ----------------------------------
            # LOCAL AI URL EXPLANATION
            # ----------------------------------
            st.divider()
            st.subheader("🤖 Local AI Security Explanation")

            if st.button("Generate AI Explanation", key="url_ai"):

                with st.spinner("Qwen is explaining the URL analysis..."):
                    try:
                        ai_findings = [
                            f"Rule-based heuristic score: {risk['score']}/100.",
                            f"Rule-based risk level: {risk['level']}.",
                        ]

                        if findings:
                            ai_findings.extend(
                                str(finding) for finding in findings
                            )
                        else:
                            ai_findings.append(
                                "No suspicious indicators were detected "
                                "by the current URL rules."
                            )

                        if ml_result is not None:
                            ai_findings.extend([
                                f"ML classification: {ml_result['prediction']}.",
                                "Uncalibrated ML phishing-class score: "
                                f"{ml_result['phishing_score'] * 100:.2f}%.",
                                f"ML decision threshold: {ml_result['threshold']}.",
                                "The ML score is not a calibrated "
                                "real-world phishing probability.",
                            ])

                            if rule_flagged != (
                                ml_result["prediction"] == "Phishing"
                            ):
                                ai_findings.append(
                                    "The rule-based and ML detectors disagree."
                                )
                        else:
                            ai_findings.append(
                                "ML analysis was unavailable. "
                                "Do not assume the URL is safe."
                            )

                        explanation = explain_security_findings(
                            input_type="URL",
                            score=risk["score"],
                            risk_level=risk["level"],
                            findings=ai_findings,
                        )

                        st.markdown(explanation)

                    except RuntimeError as exc:
                        st.error(str(exc))



    elif input_type == "Email":

        st.subheader("📧 Email Phishing Analysis")

        result = analyze_email(user_input)

        st.metric(
            "Email Heuristic Risk Score",
            f"{result['score']}/100"
        )

        if result["level"] == "High Risk":
            st.error("High Heuristic Risk")
        elif result["level"] == "Medium Risk":
            st.warning("Medium Heuristic Risk")
        else:
            st.success("Low Heuristic Risk")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Content Score",
                result["content_score"]
            )

        with col2:
            st.metric(
                "Sender Score",
                result["sender_score"]
            )

        st.subheader("Detected Indicators")

        if result["findings"]:
            for finding in result["findings"]:
                st.warning(f"⚠️ {finding}")
        else:
            st.info(
                "No suspicious indicators detected "
                "by the current rules."
            )

        if result["urls"]:
            st.subheader("Extracted URLs")

            for url in result["urls"]:
                st.code(url)

        st.divider()
        st.subheader("🔐 Security Recommendations")

        st.write(
            "• Verify the sender through an independently "
            "obtained official contact channel."
        )
        st.write(
            "• Avoid opening unexpected links or attachments."
        )
        st.write(
            "• Never share passwords or verification codes."
        )

        st.caption(
            "This analysis uses heuristic rules, not a "
            "calibrated phishing probability. "
            "Sender authentication has not been verified."
        )

        st.divider()
        st.subheader("🤖 Local AI Security Explanation")

        if st.button(
            "Generate AI Explanation",
            key="email_ai"
        ):
            with st.spinner(
                "Qwen is explaining the email findings..."
            ):
                try:
                    explanation = explain_security_findings(
                        input_type="Email",
                        score=result["score"],
                        risk_level=result["level"],
                        findings=result["findings"],
                    )

                    st.markdown(explanation)

                except RuntimeError as exc:
                    st.error(str(exc))


    elif input_type == "Text Message":

        st.subheader("📱 SMS Phishing Analysis")

        result = analyze_sms(user_input)

        st.metric(
            "SMS Heuristic Risk Score",
            f"{result['score']}/100"
        )

        if result["level"] == "High Risk":
            st.error("High Heuristic Risk")
        elif result["level"] == "Medium Risk":
            st.warning("Medium Heuristic Risk")
        else:
            st.success("Low Heuristic Risk")

        st.subheader("Detected Indicators")

        if result["findings"]:
            for finding in result["findings"]:
                st.warning(f"⚠️ {finding}")
        else:
            st.info("No indicators detected by the current rules.")

        if result["urls"]:
            st.subheader("Extracted URLs")
            for url in result["urls"]:
                st.code(url)

        st.divider()

        st.subheader("🔐 Security Recommendations")

        st.write("• Verify suspicious requests independently.")
        st.write("• Avoid opening unexpected links.")
        st.write("• Never share passwords or verification codes.")

        st.caption(
            "This analysis uses heuristic rules, not a "
            "calibrated phishing probability. "
            "A low score does not guarantee safety."
        )

        st.divider()
        st.subheader("🤖 Local AI Security Explanation")

        if st.button("Generate AI Explanation", key="sms_ai"):
            with st.spinner("Qwen is analyzing the findings..."):
                try:
                    explanation = explain_security_findings(
                        input_type="SMS",
                        score=result["score"],
                        risk_level=result["level"],
                        findings=result["findings"],
                    )

                    st.markdown(explanation)

                except RuntimeError as exc:
                    st.error(str(exc))
