"""Ad-hoc smoke test for text_repr.py against real noisy rows (not unit tests
yet -- just eyeballing behavior on real data before building on top of it).
Run: python -m src._smoke_test_text_repr
"""
from . import text_repr as tr

CASES = [
    ("S3-397034469 (Devanagari)", "रियल एग्रो प्राइवेट लिमिटेड"),
    ("Shiva Mines Pvt Ltd (Latin+suffix)", "Shiva  Mines Pvt Ltd"),
    ("Tamil business name", "சில்வர் டெக்னாலஜீஸ் புரொவிஷன் பிரைவேட் லிமிடெட்"),
    ("French SARL", "Marguerite Atelier SARL"),
    ("US PLLC", "Pediatric Medicine PLLC"),
    ("Mixed Devanagari+Latin address", "F-17, Sector 22, Noida, Gautam Buddha Nagar, उत्तर प्रदेश"),
    ("Empty", ""),
]

for label, raw in CASES:
    r = tr.build_name_repr(raw)
    print(f"--- {label} ---")
    print("normalized      :", r.normalized.encode("unicode_escape").decode("ascii"))
    print("tokens          :", [t.encode("unicode_escape").decode("ascii") for t in r.tokens])
    print("suffix_tokens   :", r.suffix_tokens)
    print("dominant_script :", r.dominant_script)
    print("skeleton        :", r.skeleton.encode("unicode_escape").decode("ascii"))
    print("num char_ngrams :", len(tr.char_ngrams(r.normalized)))  # computed on-demand, not stored
    print()

print("=== Address-specific case ===")
addr = tr.build_address_repr("4850 20, Otisco, NY")
print("normalized      :", addr.normalized)
print("leading_digits  :", addr.leading_digits)
print("digit_tokens    :", addr.digit_tokens)

print()
print("=== script_match sanity ===")
a = tr.normalize_unicode("Shiva Mines Pvt Ltd")
b = tr.normalize_unicode("Shiva Bakery")
c = tr.normalize_unicode("रियल एग्रो")
print("Latin vs Latin match:", tr.script_match(a, b), "(expect True)")
print("Latin vs Devanagari match:", tr.script_match(a, c), "(expect False)")

print()
print("=== v4 normalization / phonetic checks (asserted) ===")
CHECKS = [
    (tr.build_name_repr("Çlub FRÀNCE Mùsique").normalized, "club france musique"),
    (tr.build_name_repr("M0dern 5tudio Hatt0n").normalized, "modern studio hatton"),
    (tr.build_name_repr("21st Century 3M").normalized, "21st century 3m"),
    (tr.build_name_repr("visioncarelynn.com").normalized, "visioncarelynn"),
    (tr.build_name_repr("NULL").normalized, ""),
    (tr.build_address_repr("0023 PIERSIDE DR, Maryland").normalized, "23 pierside drive md"),
    (tr.build_address_repr("Chennai, Tamil Nadu").normalized, tr.build_address_repr("CHENNAI, தமிழ்நாடு").normalized),
    (tr.build_address_repr("N/A, <NULL>").normalized, ""),
    (tr.build_address_repr("12 Floor, Main Ct, Florida").normalized, "12 fl main ct fl"),
    (tr.phonetic_tokens(tr.build_name_repr("விஷன் லக்ஷ்மி ஃபுட்ஸ்").skeleton),
     tr.phonetic_tokens("vision laxmi foods")),
    (tr.phonetic_tokens(tr.build_name_repr("क्रिएटिव टेक्नोलॉजी प्राइवेट").skeleton),
     tr.phonetic_tokens("creative technology private")),
    (tr.build_name_repr("ಕನ್\u200cಸ್ಟ್ರಕ್ಷನ್").skeleton, "knstrkshn"),
]
for i, (got, want) in enumerate(CHECKS):
    assert got == want, f"check {i}: {got!r} != {want!r}"
print(f"all {len(CHECKS)} checks passed")
