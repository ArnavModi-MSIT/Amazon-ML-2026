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
