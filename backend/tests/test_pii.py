"""
Unit tests for the PII redaction pipeline.
"""
import pytest

from ..pii import InMemoryStore, PseudonymConfig, RedactionPipeline
from ..pii.patterns import EntityType
from ..pii.pseudonymizer import Pseudonymizer


def make_pipeline(**kwargs) -> RedactionPipeline:
    config = PseudonymConfig(
        hmac_key=b"test-secret-key-32-bytes-padding!",
        store_backend="memory",
        **kwargs,
    )
    return RedactionPipeline(config, InMemoryStore())


def make_pseudo(suffix_length: int = 4) -> Pseudonymizer:
    config = PseudonymConfig(
        hmac_key=b"test-secret-key-32-bytes-padding!",
        suffix_length=suffix_length,
    )
    return Pseudonymizer(config, InMemoryStore())


def redacted(text: str) -> tuple[str, list]:
    return make_pipeline().redact(text)


def entity_types(text: str) -> list[str]:
    _, findings = redacted(text)
    return [f.entity_type for f in findings]


# ── JWT ───────────────────────────────────────────────────────────────────────

class TestJwt:
    def test_jwt_redacted(self):
        token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4"
        out, findings = redacted(f"token={token}")
        assert token not in out
        assert "JWT" in entity_types(f"token={token}")

    def test_non_jwt_not_matched(self):
        assert "JWT" not in entity_types("hello.world.foo")


# ── PEM keys ──────────────────────────────────────────────────────────────────

class TestPemKeyDetection:
    def test_rsa_private_key_redacted(self):
        pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
               "MIIEogIBAAKCAQEArXhNUs6wmhLndodqmK4FDFedmxRTzJQ\n"
               "-----END RSA PRIVATE KEY-----")
        out, findings = redacted(pem)
        assert "MIIEog" not in out
        assert findings[0].entity_type == "PEM_KEY"

    def test_ec_private_key_redacted(self):
        pem = ("-----BEGIN EC PRIVATE KEY-----\n"
               "MHcCAQEEIIG5G0G9gJhGQb4B\n"
               "-----END EC PRIVATE KEY-----")
        _, findings = redacted(pem)
        assert findings[0].entity_type == "PEM_KEY"

    def test_openssh_private_key_redacted(self):
        pem = ("-----BEGIN OPENSSH PRIVATE KEY-----\n"
               "b3BlbnNzaC1rZXktdjEAAAAEbm9uZQAAAAAAAAAB\n"
               "-----END OPENSSH PRIVATE KEY-----")
        _, findings = redacted(pem)
        assert findings[0].entity_type == "PEM_KEY"

    def test_public_key_not_redacted(self):
        pub = ("-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkq\n-----END PUBLIC KEY-----")
        _, findings = redacted(pub)
        assert all(f.entity_type != "PEM_KEY" for f in findings)

    def test_certificate_not_redacted(self):
        cert = ("-----BEGIN CERTIFICATE-----\nMIIDazCCAlOg\n-----END CERTIFICATE-----")
        _, findings = redacted(cert)
        assert all(f.entity_type != "PEM_KEY" for f in findings)

    def test_pem_key_env_format_literal_newline(self):
        pem = r"PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
        out, findings = redacted(pem)
        assert "MIIEog" not in out
        assert any(f.entity_type == "PEM_KEY" for f in findings)

    def test_pem_key_restored_correctly(self):
        p = make_pipeline()
        pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
               "MIIEogIBAAKCAQEArXhNUs6wmhLndodqmK4F\n"
               "-----END RSA PRIVATE KEY-----")
        out, findings = p.redact(pem)
        restored = p.restore(out)
        assert "MIIEog" in restored
        assert findings[0].placeholder not in restored


# ── AI / LLM keys ─────────────────────────────────────────────────────────────

class TestAiKey:
    def test_openai_sk_key(self):
        assert "AI_KEY" in entity_types("key: sk-abcdefghijklmnopqrstuvwxyz12345")

    def test_anthropic_sk_ant_key(self):
        k = "sk-ant-api03-" + "A" * 93
        assert "AI_KEY" in entity_types(f"ANTHROPIC_KEY={k}")

    def test_openrouter_key(self):
        k = "sk-or-v1-" + "a" * 64
        assert "AI_KEY" in entity_types(k)

    def test_huggingface_key(self):
        assert "AI_KEY" in entity_types("hf_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef")

    def test_replicate_key(self):
        assert "AI_KEY" in entity_types("r8_" + "a" * 40)

    def test_short_sk_not_matched(self):
        # Too short to be a real key
        assert "AI_KEY" not in entity_types("sk-short")


# ── GitHub tokens ─────────────────────────────────────────────────────────────

class TestGhToken:
    def test_ghp_classic_pat(self):
        assert "GH_TOKEN" in entity_types("ghp_" + "A" * 36)

    def test_gho_oauth(self):
        assert "GH_TOKEN" in entity_types("gho_" + "B" * 36)

    def test_ghs_server(self):
        assert "GH_TOKEN" in entity_types("ghs_" + "C" * 36)

    def test_ghr_refresh(self):
        assert "GH_TOKEN" in entity_types("ghr_" + "D" * 36)

    def test_fine_grained_pat(self):
        tok = "github_pat_" + "A" * 22 + "_" + "B" * 59
        assert "GH_TOKEN" in entity_types(tok)


# ── AWS keys ──────────────────────────────────────────────────────────────────

class TestAwsKey:
    def test_akia_access_key(self):
        assert "AWS_KEY" in entity_types("AKIAIOSFODNN7EXAMPLE")

    def test_asia_sts_key(self):
        assert "AWS_KEY" in entity_types("ASIAIOSFODNN7EXAMPL1")

    def test_agpa_group_key(self):
        assert "AWS_KEY" in entity_types("AGPAIOSFODNN7EXAMPL2")


# ── Stripe keys ───────────────────────────────────────────────────────────────

class TestStripeKey:
    def test_live_secret_key(self):
        assert "STRIPE_KEY" in entity_types("sk_live_" + "a" * 24)

    def test_test_secret_key(self):
        assert "STRIPE_KEY" in entity_types("sk_test_" + "b" * 24)

    def test_live_publishable_key(self):
        assert "STRIPE_KEY" in entity_types("pk_live_" + "c" * 24)

    def test_test_publishable_key(self):
        assert "STRIPE_KEY" in entity_types("pk_test_" + "d" * 24)

    def test_restricted_key(self):
        assert "STRIPE_KEY" in entity_types("rk_live_" + "e" * 24)

    def test_webhook_secret(self):
        assert "STRIPE_KEY" in entity_types("whsec_" + "f" * 32)


# ── Database URLs ─────────────────────────────────────────────────────────────

class TestDbUrl:
    def test_postgres_url(self):
        out, findings = redacted("postgresql://admin:hunter2@db.internal:5432/prod")
        assert "hunter2" not in out
        assert findings[0].entity_type == "DB_URL"

    def test_mysql_url(self):
        out, findings = redacted("mysql://user:secret@localhost/mydb")
        assert "secret" not in out

    def test_mongodb_url(self):
        assert "DB_URL" in entity_types("mongodb://user:pass@cluster0.mongodb.net/db")

    def test_mongodb_srv(self):
        assert "DB_URL" in entity_types("mongodb+srv://user:pass@cluster.mongodb.net/db")

    def test_redis_url(self):
        assert "DB_URL" in entity_types("redis://default:supersecret@cache.host:6379/0")

    def test_url_without_credentials_not_redacted(self):
        assert "DB_URL" not in entity_types("connect to db.internal:5432")


# ── Slack tokens ──────────────────────────────────────────────────────────────

class TestSlackToken:
    def test_bot_token(self):
        assert "SLACK_TOKEN" in entity_types("xoxb-fakebottoken0000-fakebottoken0000")

    def test_user_token(self):
        assert "SLACK_TOKEN" in entity_types("xoxp-fakeusertoken000-fakeusertoken000")

    def test_app_token(self):
        assert "SLACK_TOKEN" in entity_types("xoxa-fakeapptoken0000-fakeapptoken0000")


# ── GCP keys ──────────────────────────────────────────────────────────────────

class TestGcpKey:
    def test_google_api_key(self):
        assert "GCP_KEY" in entity_types("AIzaSyD-9tSrke72I6e7aBcDeFgHiJkLmNoPqRs")

    def test_non_gcp_key_not_matched(self):
        assert "GCP_KEY" not in entity_types("AIzaShort")


# ── HTTP Auth headers ─────────────────────────────────────────────────────────

class TestHttpAuth:
    def test_bearer_token(self):
        out, findings = redacted("Authorization: Bearer mysupersecrettoken123")
        assert "mysupersecrettoken123" not in out
        assert "HTTP_AUTH" in [f.entity_type for f in findings]

    def test_token_scheme(self):
        out, _ = redacted("Authorization: Token abcdefghijklmnop")
        assert "abcdefghijklmnop" not in out

    def test_basic_auth(self):
        out, _ = redacted("Authorization: Basic dXNlcjpwYXNzd29yZA==")
        assert "dXNlcjpwYXNzd29yZA==" not in out

    def test_x_api_key_header(self):
        out, findings = redacted("X-Api-Key: supersecretkey12345")
        assert "supersecretkey12345" not in out
        assert any(f.entity_type == "HTTP_AUTH" for f in findings)

    def test_scheme_label_preserved(self):
        out, _ = redacted("Authorization: Bearer realtoken12345678")
        assert "Authorization" in out
        assert "Bearer" in out
        assert "realtoken12345678" not in out


# ── Generic key=value assignments ─────────────────────────────────────────────

class TestApiKey:
    def test_password_equals(self):
        out, _ = redacted("password=hunter2")
        assert "hunter2" not in out

    def test_secret_colon(self):
        out, _ = redacted("secret: my-secret-value")
        assert "my-secret-value" not in out

    def test_client_secret(self):
        out, _ = redacted("client_secret=abc123xyz456")
        assert "abc123xyz456" not in out

    def test_webhook_secret(self):
        out, _ = redacted("webhook_secret=whs_supersecret")
        assert "whs_supersecret" not in out

    def test_deploy_key(self):
        out, _ = redacted("deploy_key=deploy_secret_value")
        assert "deploy_secret_value" not in out

    def test_key_label_preserved(self):
        out, _ = redacted("api_key=topsecret123")
        assert "api_key" in out
        assert "topsecret123" not in out


# ── GitLab tokens ─────────────────────────────────────────────────────────────

class TestGitlabToken:
    def test_personal_access_token(self):
        assert "GITLAB_TOKEN" in entity_types("glpat-" + "a" * 20)

    def test_oauth_token(self):
        assert "GITLAB_TOKEN" in entity_types("gloas-" + "b" * 20)

    def test_deploy_token(self):
        assert "GITLAB_TOKEN" in entity_types("gldt-" + "c" * 20)


# ── Service-specific tokens ───────────────────────────────────────────────────

class TestServiceToken:
    def test_npm_token(self):
        assert "SERVICE_TOKEN" in entity_types("npm_" + "A" * 36)

    def test_pypi_token(self):
        assert "SERVICE_TOKEN" in entity_types("pypi-AgE" + "A" * 55)

    def test_sendgrid_key(self):
        key = "SG." + "A" * 22 + "." + "B" * 43
        assert "SERVICE_TOKEN" in entity_types(key)

    def test_twilio_account_sid(self):
        assert "SERVICE_TOKEN" in entity_types("AC" + "a" * 32)

    def test_mailgun_key(self):
        assert "SERVICE_TOKEN" in entity_types("key-" + "a" * 32)

    def test_mailchimp_key(self):
        assert "SERVICE_TOKEN" in entity_types("a" * 32 + "-us12")

    def test_shopify_access_token(self):
        assert "SERVICE_TOKEN" in entity_types("shpat_" + "a" * 32)

    def test_shopify_shared_secret(self):
        assert "SERVICE_TOKEN" in entity_types("shpss_" + "b" * 32)

    def test_telegram_bot_token(self):
        assert "SERVICE_TOKEN" in entity_types("123456789:" + "A" * 35)

    def test_docker_pat(self):
        assert "SERVICE_TOKEN" in entity_types("dckr_pat_" + "A" * 43)

    def test_digitalocean_token(self):
        assert "SERVICE_TOKEN" in entity_types("dop_v1_" + "a" * 64)

    def test_airtable_pat(self):
        assert "SERVICE_TOKEN" in entity_types("pat" + "A" * 14 + "." + "B" * 64)

    def test_linear_api_key(self):
        assert "SERVICE_TOKEN" in entity_types("lin_api_" + "A" * 40)


# ── Generic bare token ────────────────────────────────────────────────────────

class TestGenericToken:
    def test_deepseek_style_key(self):
        assert "GENERIC_TOKEN" in entity_types("VU45MrFmcik2COjYnHTCjbzcd8eXQ7kJXyoGFFZd")

    def test_all_lowercase_hex_not_matched(self):
        # git SHA — all lowercase, no uppercase → should NOT match
        assert "GENERIC_TOKEN" not in entity_types("a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9")

    def test_too_short_not_matched(self):
        assert "GENERIC_TOKEN" not in entity_types("Ab1cDe2f")

    def test_no_digit_not_matched(self):
        assert "GENERIC_TOKEN" not in entity_types("AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh")

    def test_bare_token_in_sentence(self):
        out, findings = redacted("My api key is VU45MrFmcik2COjYnHTCjbzcd8eXQ7kJXyoGFFZd please dont leak it")
        assert "VU45MrFmcik2COjYnHTCjbzcd8eXQ7kJXyoGFFZd" not in out


# ── Google OAuth ──────────────────────────────────────────────────────────────

class TestOauthToken:
    def test_google_oauth_token(self):
        tok = "ya29." + "A" * 50
        assert "OAUTH_TOKEN" in entity_types(tok)


# ── SSN ───────────────────────────────────────────────────────────────────────

class TestSsn:
    def test_ssn_redacted(self):
        out, findings = redacted("SSN: 123-45-6789")
        assert "123-45-6789" not in out
        assert findings[0].entity_type == "SSN"

    def test_version_string_not_ssn(self):
        assert "SSN" not in entity_types("version 1.2.3456")


# ── Email ─────────────────────────────────────────────────────────────────────

class TestEmailDetection:
    def test_real_email_redacted(self):
        out, findings = redacted("Send invoice to finance@acme.com")
        assert "finance@acme.com" not in out
        assert findings[0].entity_type == "EMAIL"

    def test_dummy_domain_not_redacted(self):
        assert len(make_pipeline().redact("alice@example.org")[1]) == 0

    def test_excluded_domains(self):
        for domain in ("example", "test", "placeholder", "dummy", "sample", "localhost", "foo", "bar"):
            _, findings = redacted(f"user@{domain}.com")
            assert len(findings) == 0

    def test_multiple_emails_distinct_placeholders(self):
        _, findings = redacted("alice@corp.io and bob@corp.io")
        assert len(findings) == 2
        assert len({f.placeholder for f in findings}) == 2

    def test_placeholder_is_deterministic(self):
        _, f1 = redacted("foo@corp.io")
        _, f2 = redacted("foo@corp.io")
        assert f1[0].placeholder == f2[0].placeholder


# ── Phone ─────────────────────────────────────────────────────────────────────

class TestPhoneDetection:
    def test_us_phone_dashes(self):
        _, findings = redacted("Call me at 555-867-5309")
        assert findings[0].entity_type == "PHONE"

    def test_international_phone(self):
        _, findings = redacted("+1 800 555 1234")
        assert findings[0].entity_type == "PHONE"

    def test_port_number_not_phone(self):
        _, findings = redacted("Server listens on port 8080")
        assert all(f.entity_type != "PHONE" for f in findings)


# ── Overlap resolution ────────────────────────────────────────────────────────

class TestOverlapResolution:
    def test_db_url_wins_over_email_fragment(self):
        _, findings = redacted("postgresql://user:password123@host/db")
        assert any(f.entity_type == "DB_URL" for f in findings)
        assert len(findings) == 1

    def test_jwt_wins_over_generic_token(self):
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.SflKxwRJSMeKKF2QT4fwpMeJf36P"
        types = entity_types(jwt)
        assert "JWT" in types
        assert "GENERIC_TOKEN" not in types

    def test_stripe_key_wins_over_generic_token(self):
        types = entity_types("sk_live_" + "Aa1" * 10)
        assert "STRIPE_KEY" in types
        assert "GENERIC_TOKEN" not in types


# ── Placeholder semantics ─────────────────────────────────────────────────────

class TestPlaceholderSemantics:
    def test_email_placeholder_format(self):
        _, f = redacted("admin@company.com")
        assert f[0].placeholder.startswith("EMAIL_")

    def test_phone_placeholder_format(self):
        _, f = redacted("(555) 867-5309")
        assert f[0].placeholder.startswith("PHONE_")

    def test_db_url_placeholder_format(self):
        _, f = redacted("postgres://u:pass@host/db")
        assert f[0].placeholder.startswith("DB_URL_")

    def test_stripe_placeholder_format(self):
        _, f = redacted("sk_live_" + "a" * 24)
        assert f[0].placeholder.startswith("STRIPE_KEY_")

    def test_generic_token_placeholder_format(self):
        _, f = redacted("VU45MrFmcik2COjYnHTCjbzcd8eXQ7kJXyoGFFZd")
        assert f[0].placeholder.startswith("GENERIC_TOKEN_")


# ── Pseudonymizer unit tests ──────────────────────────────────────────────────

class TestPseudonymizer:
    def test_deterministic(self):
        p = make_pseudo()
        assert p.pseudonymize(EntityType.EMAIL, "alice@corp.io") == \
               p.pseudonymize(EntityType.EMAIL, "alice@corp.io")

    def test_different_values_different_placeholders(self):
        p = make_pseudo()
        assert p.pseudonymize(EntityType.EMAIL, "alice@corp.io") != \
               p.pseudonymize(EntityType.EMAIL, "bob@corp.io")

    def test_placeholder_format(self):
        p = make_pseudo()
        ph = p.pseudonymize(EntityType.EMAIL, "user@corp.io")
        parts = ph.split("_")
        assert parts[0] == "EMAIL"
        int(parts[-1], 16)  # suffix must be valid hex

    def test_suffix_is_uppercase_hex(self):
        p = make_pseudo()
        ph = p.pseudonymize(EntityType.DB_URL, "postgres://u:s@h/db")
        suffix = ph.split("_")[-1]
        assert suffix == suffix.upper()
        int(suffix, 16)

    def test_custom_suffix_length(self):
        p = make_pseudo(suffix_length=6)
        ph = p.pseudonymize(EntityType.EMAIL, "cto@bigcorp.com")
        assert len(ph.split("_")[1]) == 6

    def test_collision_gets_counter(self):
        store = InMemoryStore()
        config = PseudonymConfig(hmac_key=b"test-secret-key-32-bytes-padding!")
        p = Pseudonymizer(config, store)
        real_ph = p.pseudonymize(EntityType.EMAIL, "first@corp.io")
        store.put(real_ph, "different@corp.io")
        store.put("different@corp.io", real_ph)
        candidate = p._resolve_collision(real_ph, "another@corp.io")
        assert candidate == f"{real_ph}_1"


# ── Context logging ───────────────────────────────────────────────────────────

class TestContextLogging:
    def test_context_in_findings(self):
        _, findings = redacted("Hello user@corp.io, how are you?")
        ctx = findings[0].context
        assert "Hello" in ctx
        assert "how are you" in ctx


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_requests_counted(self):
        p = make_pipeline()
        p.redact("no pii here")
        p.redact("email: boss@corp.io")
        m = p.metrics.snapshot(store_size=p.store.size())
        assert m["requests_total"] == 2
        assert m["redacted_requests"] == 1
        assert m["findings_total"] == 1

    def test_clean_request_not_counted_as_redacted(self):
        p = make_pipeline()
        p.redact("just normal text")
        m = p.metrics.snapshot(store_size=p.store.size())
        assert m["redacted_requests"] == 0
