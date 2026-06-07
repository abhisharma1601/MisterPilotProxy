"""
Unit tests for the PII redaction pipeline.

Covers the 3 entity types in MisterPilot's pattern registry:
  EMAIL, PHONE, DB_URL
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


# --------------------------------------------------------------------------- #
# Email detection                                                              #
# --------------------------------------------------------------------------- #

class TestEmailDetection:
    def test_dummy_domain_not_redacted(self):
        p = make_pipeline()
        out, findings = p.redact("Contact alice@example.org for details")
        assert "alice@example.org" in out
        assert len(findings) == 0

    def test_real_email_redacted(self):
        p = make_pipeline()
        out, findings = p.redact("Send invoice to finance@acme.com")
        assert "finance@acme.com" not in out
        assert len(findings) == 1
        assert findings[0].entity_type == "EMAIL"
        assert "EMAIL_" in findings[0].placeholder

    def test_placeholder_is_deterministic(self):
        p = make_pipeline()
        _, f1 = p.redact("foo@corp.io")
        _, f2 = p.redact("foo@corp.io")
        assert f1[0].placeholder == f2[0].placeholder

    def test_multiple_emails_get_distinct_placeholders(self):
        p = make_pipeline()
        _, findings = p.redact("alice@corp.io and bob@corp.io are cc'd")
        assert len(findings) == 2
        assert len({f.placeholder for f in findings}) == 2

    def test_excluded_domains(self):
        for domain in ("example", "test", "placeholder", "dummy", "sample", "localhost", "foo", "bar"):
            p = make_pipeline()
            out, findings = p.redact(f"user@{domain}.com")
            assert len(findings) == 0, f"Should not redact @{domain}.com"


# --------------------------------------------------------------------------- #
# Phone detection                                                              #
# --------------------------------------------------------------------------- #

class TestPhoneDetection:
    def test_us_phone_with_dashes(self):
        p = make_pipeline()
        out, findings = p.redact("Call me at 555-867-5309")
        assert "555-867-5309" not in out
        assert len(findings) == 1
        assert findings[0].entity_type == "PHONE"

    def test_international_phone(self):
        p = make_pipeline()
        out, findings = p.redact("Reach us at +1 800 555 1234")
        assert findings[0].entity_type == "PHONE"

    def test_port_number_not_redacted(self):
        p = make_pipeline()
        out, findings = p.redact("Server listens on port 8080")
        phone_findings = [f for f in findings if f.entity_type == "PHONE"]
        assert len(phone_findings) == 0


# --------------------------------------------------------------------------- #
# DB URL detection                                                             #
# --------------------------------------------------------------------------- #

class TestDbUrlDetection:
    def test_postgres_url_with_password(self):
        p = make_pipeline()
        url = "postgresql://admin:hunter2@db.internal:5432/prod"
        out, findings = p.redact(url)
        assert "hunter2" not in out
        assert findings[0].entity_type == "DB_URL"

    def test_mysql_url_redacted(self):
        p = make_pipeline()
        out, findings = p.redact("mysql://user:secret@localhost/mydb")
        assert "secret" not in out
        assert findings[0].entity_type == "DB_URL"

    def test_url_without_credentials_not_redacted(self):
        p = make_pipeline()
        out, findings = p.redact("connect to db.internal:5432")
        db_findings = [f for f in findings if f.entity_type == "DB_URL"]
        assert len(db_findings) == 0


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# PEM private key detection                                                    #
# --------------------------------------------------------------------------- #

class TestPemKeyDetection:
    def test_rsa_private_key_redacted(self):
        p = make_pipeline()
        pem = """-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEArXhNUs6wmhLndodqmK4FDFedmxRTzJQ+Hsjo4Od077d6x4vq
VlgofAqODiMdoWsPRa7Gl5H0L6T1wlinWNdKgJQR2vQuFPcUZLUmf/9biDPTRbEc
gJPFuFmQLmMiALBM1IU7iisGnj+G5iN7+mPoAZceWfuMypJ9sWTSVzJQ+pC+TSN1
5oIaRqlvSOaqEhV1vMhV4W2lK/LqLdugu7N7v1vHdBX+Yg0i7bK6bFPys0F/kcGw
-----END RSA PRIVATE KEY-----"""
        out, findings = p.redact(pem)
        assert "MIIEog" not in out
        assert len(findings) == 1
        assert findings[0].entity_type == "PEM_KEY"
        assert findings[0].placeholder.startswith("PEM_KEY_")

    def test_ec_private_key_redacted(self):
        p = make_pipeline()
        pem = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIIG5G0G9gJhGQb4B5G0G9gJhGQb4B5G0G9gJhGQb4B5GoAoGCCqGSM49
AwEHoUQDQgAE5G0G9gJhGQb4B5G0G9gJhGQb4B5G0G9gJhGQb4B5G0G9gJhGQb4
-----END EC PRIVATE KEY-----"""
        out, findings = p.redact(pem)
        assert len(findings) == 1
        assert findings[0].entity_type == "PEM_KEY"

    def test_openssh_private_key_redacted(self):
        p = make_pipeline()
        pem = """-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
NhAAAAAwEAAQAAAYEArXhNUs6wmhLndodqmK4FDFedmxRTzJQ+Hsjo4Od077d6x4vq
-----END OPENSSH PRIVATE KEY-----"""
        out, findings = p.redact(pem)
        assert len(findings) == 1
        assert findings[0].entity_type == "PEM_KEY"

    def test_public_key_not_redacted(self):
        p = make_pipeline()
        pub = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEArXhNUs6w
-----END PUBLIC KEY-----"""
        out, findings = p.redact(pub)
        pem_findings = [f for f in findings if f.entity_type == "PEM_KEY"]
        assert len(pem_findings) == 0, "Public keys should not be redacted"

    def test_certificate_not_redacted(self):
        p = make_pipeline()
        cert = """-----BEGIN CERTIFICATE-----
MIIDazCCAlOgAwIBAgIUJ...
-----END CERTIFICATE-----"""
        out, findings = p.redact(cert)
        pem_findings = [f for f in findings if f.entity_type == "PEM_KEY"]
        assert len(pem_findings) == 0, "Certificates should not be redacted"

    def test_pem_key_placeholder_format(self):
        p = make_pipeline()
        pem = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQC
-----END PRIVATE KEY-----"""
        _, f = p.redact(pem)
        assert f[0].placeholder.startswith("PEM_KEY_")
        assert f[0].placeholder.count("_") >= 1

    def test_pem_key_restored_correctly(self):
        p = make_pipeline()
        pem = """-----BEGIN RSA PRIVATE KEY-----
MIIEogIBAAKCAQEArXhNUs6wmhLndodqmK4FDFedmxRTzJQ
-----END RSA PRIVATE KEY-----"""
        out, findings = p.redact(pem)
        placeholder = findings[0].placeholder
        restored = p.restore(out)
        assert "MIIEog" in restored
        assert placeholder not in restored

# Overlap resolution                                                           #
# --------------------------------------------------------------------------- #

class TestOverlapResolution:
    def test_db_url_wins_over_email_fragment(self):
        """DB_URL (priority=30) should consume the whole URL as one finding."""
        p = make_pipeline()
        url = "postgresql://user:password123@host/db"
        _, findings = p.redact(url)
        assert any(f.entity_type == "DB_URL" for f in findings)
        assert len(findings) == 1


# --------------------------------------------------------------------------- #
# Placeholder semantics                                                        #
# --------------------------------------------------------------------------- #

class TestPlaceholderSemantics:
    def test_email_placeholder_format(self):
        p = make_pipeline()
        _, f = p.redact("admin@company.com")
        assert f[0].placeholder.startswith("EMAIL_")

    def test_phone_placeholder_format(self):
        p = make_pipeline()
        _, f = p.redact("(555) 867-5309")
        assert f[0].placeholder.startswith("PHONE_")

    def test_db_url_placeholder_format(self):
        p = make_pipeline()
        _, f = p.redact("postgres://u:pass@host/db")
        assert f[0].placeholder.startswith("DB_URL_")


# --------------------------------------------------------------------------- #
# Pseudonymizer unit tests                                                     #
# --------------------------------------------------------------------------- #

class TestPseudonymizer:
    def test_deterministic(self):
        p = make_pseudo()
        a = p.pseudonymize(EntityType.EMAIL, "alice@corp.io")
        b = p.pseudonymize(EntityType.EMAIL, "alice@corp.io")
        assert a == b

    def test_different_values_different_placeholders(self):
        p = make_pseudo()
        a = p.pseudonymize(EntityType.EMAIL, "alice@corp.io")
        b = p.pseudonymize(EntityType.EMAIL, "bob@corp.io")
        assert a != b

    def test_placeholder_format(self):
        p = make_pseudo()
        ph = p.pseudonymize(EntityType.EMAIL, "user@corp.io")
        parts = ph.split("_")
        assert parts[0] == "EMAIL"
        suffix = parts[-1]
        assert len(suffix) == 4
        int(suffix, 16)  # must be valid hex

    def test_suffix_is_uppercase_hex(self):
        p = make_pseudo()
        ph = p.pseudonymize(EntityType.DB_URL, "postgres://u:s@h/db")
        suffix = ph.split("_")[-1]
        int(suffix, 16)  # valid hex
        assert suffix == suffix.upper()

    def test_custom_suffix_length(self):
        p = make_pseudo(suffix_length=6)
        ph = p.pseudonymize(EntityType.EMAIL, "cto@bigcorp.com")
        suffix = ph.split("_")[1]
        assert len(suffix) == 6

    def test_collision_gets_counter(self):
        store = InMemoryStore()
        config = PseudonymConfig(hmac_key=b"test-secret-key-32-bytes-padding!")
        p = Pseudonymizer(config, store)
        real_ph = p.pseudonymize(EntityType.EMAIL, "first@corp.io")
        store.put(real_ph, "different@corp.io")
        store.put("different@corp.io", real_ph)
        candidate = p._resolve_collision(real_ph, "another@corp.io")
        assert candidate == f"{real_ph}_1"


# --------------------------------------------------------------------------- #
# Context logging                                                              #
# --------------------------------------------------------------------------- #

class TestContextLogging:
    def test_context_in_findings(self):
        p = make_pipeline()
        _, findings = p.redact("Hello user@corp.io, how are you?")
        assert len(findings) == 1
        ctx = findings[0].context
        assert "Hello" in ctx
        assert "how are you" in ctx


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

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
