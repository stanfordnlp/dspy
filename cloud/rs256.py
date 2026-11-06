"""
lm15.cloud.rs256 — RSASSA-PKCS1-v1_5 / SHA-256 signing, standard library only.

Used for the two JWT assertions in spec/auth.md AUTH-11 (``jwt-rs256``):
Google service accounts (RS256 is "the only signing algorithm supported",
gcp-service-account-oauth.md:318-333) and Microsoft Entra certificate
credentials (azure-identity signs RS256, certificate.py:20).

What is implemented, and only that:

- PEM → DER for ``PRIVATE KEY`` (PKCS#8), ``RSA PRIVATE KEY`` (PKCS#1),
  and ``CERTIFICATE`` blocks;
- a minimal DER reader (SEQUENCE, INTEGER, OCTET STRING, NULL, OID);
- EMSA-PKCS1-v1_5 encoding with the SHA-256 DigestInfo prefix (RFC 8017
  §9.2) and the RSASP1 primitive ``pow(m, d, n)``;
- the compact JWS serialization lm15 pins (compact JSON, keys in the
  caller's order, base64url without padding).

Stated trade-offs (changes/2026-09-03-cloud-hosts.md):

- ``pow`` on Python integers is not constant-time and has no RSA blinding.
  This implementation is not hardened against timing attacks. Use an
  external credential provider when that threat matters. Ports with a
  platform RSA (Go ``crypto/rsa``, Node ``crypto``) use it.
- Encrypted PEM and PKCS#12 are not parsed: the error names
  ``openssl pkcs12 -nodes`` / ``openssl pkey``.
- The Chinese-remainder speed-up is not used: one 2048-bit signature is
  about 75 ms in pure Python (measured 2026-09-03), once per token
  lifetime of an hour.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from ..errors import NotConfiguredError

__all__ = ["RsaPrivateKey", "load_private_key", "certificate_der", "sign_pkcs1v15_sha256", "jwt_encode", "b64url"]

# DigestInfo prefix for SHA-256 (RFC 8017 §9.2, note 1).
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")
_RSA_OID = bytes.fromhex("2a864886f70d010101")  # 1.2.840.113549.1.1.1


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


# ─── PEM / DER ───────────────────────────────────────────────────────


def _pem_block(text: str, label: str) -> bytes | None:
    head, tail = f"-----BEGIN {label}-----", f"-----END {label}-----"
    start = text.find(head)
    if start < 0:
        return None
    end = text.find(tail, start)
    if end < 0:
        raise ValueError(f"PEM block {label!r} has no END line")
    body = "".join(text[start + len(head):end].split())
    return base64.b64decode(body)


def _der_read(data: bytes, pos: int) -> tuple[int, bytes, int]:
    """(tag, value, next_pos) for the TLV at ``pos``."""
    if pos >= len(data):
        raise ValueError("DER: unexpected end")
    tag = data[pos]
    pos += 1
    if pos >= len(data):
        raise ValueError("DER: truncated length")
    length = data[pos]
    pos += 1
    if length & 0x80:
        count = length & 0x7F
        if count == 0 or count > 4:
            raise ValueError("DER: unsupported length form")
        if pos + count > len(data):
            raise ValueError("DER: truncated length")
        length = int.from_bytes(data[pos:pos + count], "big")
        pos += count
    value = data[pos:pos + length]
    if len(value) != length:
        raise ValueError("DER: truncated value")
    return tag, value, pos + length


def _der_sequence(data: bytes) -> list[tuple[int, bytes]]:
    tag, value, end = _der_read(data, 0)
    if tag != 0x30:
        raise ValueError(f"DER: expected SEQUENCE, got tag {tag:#x}")
    if end != len(data):
        raise ValueError("DER: trailing bytes after SEQUENCE")
    return _der_items(value)


def _der_items(value: bytes) -> list[tuple[int, bytes]]:
    """The TLVs inside a constructed value (a SEQUENCE's contents)."""
    items: list[tuple[int, bytes]] = []
    pos = 0
    while pos < len(value):
        item_tag, item_value, pos = _der_read(value, pos)
        items.append((item_tag, item_value))
    return items


def _der_int(item: tuple[int, bytes]) -> int:
    tag, value = item
    if tag != 0x02:
        raise ValueError(f"DER: expected INTEGER, got tag {tag:#x}")
    return int.from_bytes(value, "big", signed=False)


@dataclass(frozen=True, slots=True)
class RsaPrivateKey:
    n: int
    e: int
    d: int

    def __repr__(self) -> str:  # AUTH-5
        return f"RsaPrivateKey(bits={self.n.bit_length()}, <redacted>)"

    @property
    def size_bytes(self) -> int:
        return (self.n.bit_length() + 7) // 8


def _rsa_from_pkcs1(der: bytes) -> RsaPrivateKey:
    items = _der_sequence(der)
    # RSAPrivateKey ::= SEQUENCE { version, n, e, d, p, q, dP, dQ, qInv }
    if len(items) < 4:
        raise ValueError("PKCS#1 RSAPrivateKey: too few fields")
    return RsaPrivateKey(n=_der_int(items[1]), e=_der_int(items[2]), d=_der_int(items[3]))


def _rsa_from_pkcs8(der: bytes) -> RsaPrivateKey:
    items = _der_sequence(der)
    # PrivateKeyInfo ::= SEQUENCE { version, AlgorithmIdentifier, OCTET STRING privateKey }
    if len(items) < 3 or items[1][0] != 0x30 or items[2][0] != 0x04:
        raise ValueError("PKCS#8 PrivateKeyInfo: unexpected shape")
    alg = _der_items(items[1][1])
    if not alg or alg[0][0] != 0x06 or alg[0][1] != _RSA_OID:
        raise ValueError("PKCS#8: not an RSA key (only rsaEncryption is supported)")
    return _rsa_from_pkcs1(items[2][1])


def load_private_key(pem: str) -> RsaPrivateKey:
    """Parse an unencrypted RSA private key from PEM (PKCS#8 or PKCS#1)."""
    if "ENCRYPTED PRIVATE KEY" in pem or "Proc-Type: 4,ENCRYPTED" in pem:
        raise NotConfiguredError(
            "encrypted private keys are not supported; decrypt it first: "
            "openssl pkey -in key.pem -out key-plain.pem",
            credential_hint="openssl pkey -in key.pem -out key-plain.pem",
        )
    der = _pem_block(pem, "PRIVATE KEY")
    if der is not None:
        return _rsa_from_pkcs8(der)
    der = _pem_block(pem, "RSA PRIVATE KEY")
    if der is not None:
        return _rsa_from_pkcs1(der)
    if "BEGIN EC PRIVATE KEY" in pem:
        raise NotConfiguredError("EC private keys are not supported (RS256 needs an RSA key)")
    raise NotConfiguredError(
        "no PEM private key found; PKCS#12 (.pfx/.p12) is not parsed — convert with: "
        "openssl pkcs12 -in cert.pfx -nodes -out cert.pem",
        credential_hint="openssl pkcs12 -in cert.pfx -nodes -out cert.pem",
    )


def certificate_der(pem: str) -> bytes:
    """The DER bytes of the first CERTIFICATE block (for ``x5t`` thumbprints)."""
    der = _pem_block(pem, "CERTIFICATE")
    if der is None:
        raise NotConfiguredError("no PEM CERTIFICATE block found")
    return der


# ─── Signing ─────────────────────────────────────────────────────────


def sign_pkcs1v15_sha256(key: RsaPrivateKey, message: bytes) -> bytes:
    """RSASSA-PKCS1-v1_5-SIGN with SHA-256 (RFC 8017 §8.2.1)."""
    k = key.size_bytes
    digest = _SHA256_DIGEST_INFO + hashlib.sha256(message).digest()
    if k < len(digest) + 11:
        raise ValueError("RSA key too small for SHA-256 PKCS#1 v1.5")
    padded = b"\x00\x01" + b"\xff" * (k - len(digest) - 3) + b"\x00" + digest
    m = int.from_bytes(padded, "big")
    s = pow(m, key.d, key.n)
    return s.to_bytes(k, "big")


def jwt_encode(header: Mapping[str, Any], payload: Mapping[str, Any], key: RsaPrivateKey) -> str:
    """Compact JWS: lm15 pins compact JSON with keys in the given order and
    unpadded base64url (a canonical fact so four ports emit identical bytes;
    RFC 7519 servers accept any valid JSON)."""
    head = b64url(json.dumps(dict(header), separators=(",", ":")).encode("utf-8"))
    body = b64url(json.dumps(dict(payload), separators=(",", ":")).encode("utf-8"))
    signing_input = f"{head}.{body}".encode("ascii")
    return f"{head}.{body}.{b64url(sign_pkcs1v15_sha256(key, signing_input))}"
