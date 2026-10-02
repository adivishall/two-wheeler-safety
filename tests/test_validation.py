"""Input-validation + rate-limiter unit tests (pure, deterministic)."""

from modules.validation import (
    IMAGE_EXTENSIONS,
    RateLimiter,
    is_allowed_extension,
    safe_evidence_name,
    safe_extension,
    sniff_image,
    validate_amount,
    validate_plate,
    validate_violation,
)


def test_extension_allowlist():
    assert is_allowed_extension("photo.JPG", IMAGE_EXTENSIONS)
    assert is_allowed_extension("a.png", IMAGE_EXTENSIONS)
    assert not is_allowed_extension("a.txt", IMAGE_EXTENSIONS)
    assert not is_allowed_extension("noext", IMAGE_EXTENSIONS)


def test_safe_extension_falls_back_for_bad_input():
    assert safe_extension("x.png", IMAGE_EXTENSIONS, ".jpg") == ".png"
    assert safe_extension("x.exe", IMAGE_EXTENSIONS, ".jpg") == ".jpg"
    assert safe_extension("", IMAGE_EXTENSIONS, ".jpg") == ".jpg"


def test_sniff_image_by_magic_bytes():
    assert sniff_image(b"\xff\xd8\xff\xe0somejpegdata")   # JPEG
    assert sniff_image(b"\x89PNG\r\n\x1a\n....")           # PNG
    assert not sniff_image(b"not an image at all")
    assert not sniff_image(b"")


def test_safe_evidence_name_blocks_traversal_and_bad_types():
    assert safe_evidence_name("evidence/ok.jpg") == "ok.jpg"
    assert safe_evidence_name("a/b/c.png") == "c.png"
    assert safe_evidence_name("../../etc/passwd") is None      # no image ext
    assert safe_evidence_name("../../etc/passwd.jpg") == "passwd.jpg"  # basename only
    assert safe_evidence_name("note.txt") is None
    assert safe_evidence_name("") is None
    assert safe_evidence_name(None) is None
    # A name that reduces to "." / ".." is rejected, and an embedded NUL is
    # stripped out even with an allowed extension (defense-in-depth guards).
    assert safe_evidence_name("foo/.") is None
    assert safe_evidence_name("..") is None
    assert safe_evidence_name("a\x00.jpg") is None


def test_safe_evidence_name_rejects_markup_and_quote_characters():
    # The stored name is echoed back by /get_fines and /api/violations and
    # rendered into an <img src="..."> by the dashboard. A quote or angle
    # bracket in it is an attribute break-out (stored XSS), so only the
    # characters the pipeline itself ever writes are accepted.
    for hostile in (
        'x" onerror="alert(document.domain)" a=".jpg',
        "<svg onload=alert(1)>.jpg",
        "it's.jpg",
        "a b.jpg",
        "a\n.jpg",
        "a&b.jpg",
        "a`b.jpg",
    ):
        assert safe_evidence_name(hostile) is None, hostile
    # Names the pipeline / demo / CLI actually produce still pass.
    for ok in (
        "MH12AB1234_no_helmet_f12_ab12cd34ef_annotated.jpg",
        "demo_MH02DL4596.jpg",
        "MH02DL4596_1786603781.jpg",
        "plate-1.v2.jpeg",
    ):
        assert safe_evidence_name("evidence/" + ok) == ok


def test_validate_plate():
    assert validate_plate("  mh12ab1234 ") == "MH12AB1234"
    assert validate_plate("!!") is None       # empty after normalize
    assert validate_plate("A") is None        # too short
    assert validate_plate("X" * 20) is None   # too long


def test_validate_violation_allows_unknown_but_safe_slugs():
    assert validate_violation("no_helmet")
    assert validate_violation("some_unknown_violation")  # unknown but safe
    assert not validate_violation("")
    assert not validate_violation("drop table;")  # unsafe chars
    assert not validate_violation("x" * 100)      # too long


def test_validate_amount():
    assert validate_amount(500)
    assert not validate_amount(0)
    assert not validate_amount(-5)
    assert not validate_amount(2_000_000)
    assert not validate_amount("500")


def test_rate_limiter_sliding_window():
    rl = RateLimiter(max_requests=2, window_s=10.0)
    assert rl.allow("ip", now=0.0)
    assert rl.allow("ip", now=1.0)
    assert not rl.allow("ip", now=2.0)   # third within window -> blocked
    assert rl.allow("ip", now=11.5)      # first two aged out -> allowed again
    assert rl.allow("other", now=2.0)    # a different key is independent
