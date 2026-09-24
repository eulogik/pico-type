# Gates: ARTH full verification pass (2026-09-23)

OWNS: (verification only; fixes land in tracked source as findings require)

Scope: prove every shipped claim, every fix, and every guard is correct before proceeding (merge/hypercube/semantic-tier decisions).

- [x] G1: lint + full test suite green
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/ruff check . && .venv/bin/python -m pytest tests/ -q"
  EXPECT: 40 passed
  EVIDENCE: automatic-evidence=v1; definition-sha256=be2e161b51fec1b6b0264f7012ae5b21a99393d229e3fce079621c8fdb25ee3e; exit=0; EXPECT=matched; output-sha256=387f488b5893314e4d6fca36b7a06ba089e241605a8dcce4bb67cdba62bdce5b; output-bytes=2908; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G2: manifest determinism (all roots, incl. amendment #3)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python -c \"from model.pico_type.arth_data import verify_manifest; raise SystemExit(0 if verify_manifest() else 1)\""
  EXPECT: manifest verify: PASS
  EVIDENCE: automatic-evidence=v1; definition-sha256=379bdde4d170bbe605c9ca27e12ce0b4ef415de89b96bbe326b56aa63704a12e; exit=0; EXPECT=matched; output-sha256=a110e0a8228d049241e06b43fa9411d53c565d15b8e74bd1e4d3fc3c1f62baca; output-bytes=22; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G3: shipped-ckpt battery reproduces documented numbers
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py battery"
  EXPECT: BATTERY CLAIMS VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=9ca6baec35885a75da8bef98e5cf66ddb41de3e4128d60dde5a9d276c3baf8ec; exit=0; EXPECT=matched; output-sha256=a7005e7f5bd6e5e6f92378897778db70db29b077543a5b93ce0b3671c8a54952; output-bytes=580; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G4: ToxicChat gate reproduces on shipped ckpt
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py toxicchat"
  EXPECT: TOXICCHAT CLAIMS VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=3b8b4fbca1d958cbf4bee2d4d2f09baf40d7f441761b8f6d9a71992d42611892; exit=0; EXPECT=matched; output-sha256=c1cd9f8484ecf65617e04f0290fd0bdb55ef60d344deeb442816db8032139c54; output-bytes=226; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G5: external hand-labeled audit reproduces 0.0589
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py audit"
  EXPECT: AUDIT CLAIMS VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=5133128287a7bea2da9e8c5686a1b7b7d2052d700af66dae4742eedf88be913d; exit=0; EXPECT=matched; output-sha256=558a078a59027633c9ea3e90ef2c011e361173a9046812a326ba8de0a8a4b866; output-bytes=1144; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G6: AG/SST/Enron measured FAIL values reproduce (documented honest miss)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py gates"
  EXPECT: GATE MEASUREMENTS VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=e6b08fe68f8312003a7d8148271d20fa2d9ea8b9b0f523c1ac97120957ff6161; exit=0; EXPECT=matched; output-sha256=5dc9077ee8d78edfcd85386dccc79beccfa13f0029a5a5ed0e2a0fcb9d714695; output-bytes=225; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G7: tracked ONNX artifacts match documented size/IR/opset and are single-file
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py onnx_meta"
  EXPECT: ONNX METADATA VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=68b500662080b1d6e19abf8218042f0eab426a16758c1b60b3d4530cee03e8ea; exit=0; EXPECT=matched; output-sha256=7f0dbed48a86560ea1ff152e97462d7f1fa4690904622ff3fd69fd7193d2912f; output-bytes=122; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G8: ONNX re-export + 5-check verify passes (reproducibility)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py onnx_repro"
  EXPECT: ONNX REPRODUCIBILITY VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=ede47ffac0ca1776af560a36c458a1f15693444cf5dee980d10f2d6da0be7c91; exit=0; EXPECT=matched; output-sha256=fa2708974320ab590669c71f705b18a57a26344107ef36d64cbc32490fe800d2; output-bytes=375; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G9: cal regression test has teeth (negative control: fails on reintroduced bug)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py cal_teeth"
  EXPECT: CAL REGRESSION TEST HAS TEETH
  EVIDENCE: automatic-evidence=v1; definition-sha256=2777c665d274ff90070502c0108b1367353fc3b565b1cb7e7e17b7240a6822cb; exit=0; EXPECT=matched; output-sha256=b9dfe25166ae0cd1ff29759a8168aec61de77f8ab8c1e586664e2bf6efb349ef; output-bytes=921; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G10: credential/PII guards have teeth (negative control)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py guard_teeth"
  EXPECT: GUARDS HAVE TEETH
  EVIDENCE: automatic-evidence=v1; definition-sha256=949ed5367c01442cdadc41de47e8098f8f65d60f297861a382e145f0a7e4d1e6; exit=0; EXPECT=matched; output-sha256=8a6242d8dd8013377f77e3bdbf06b562c4aeb7ee64890d44ea4ff97958714471; output-bytes=83; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G11: purity — audit-vs-train and eval-vs-train overlaps are zero
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py purity"
  EXPECT: PURITY VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=352d9142aff3ccb208b2b74a68886747923670c6034d95de9362b8f32872c84c; exit=0; EXPECT=matched; output-sha256=9d4a827f1983c91c74ff0e280db4a4f29590f745d5c2b933b8aa02a6e4931f94; output-bytes=154; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G12: docs carry canonical numbers, no stale ckpt/gate claims, append-only history intact
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py docs"
  EXPECT: DOCS VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=8efd41e577280e3a28568c5577c8a412d3c7426fd54934afa7ea6f57198956df; exit=0; EXPECT=matched; output-sha256=c36120580a327ccbd6bb92fd93cc8fa802cd279e9d3b3ccc52219bef18db279b; output-bytes=89; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G13: shipped ckpt integrity (strict load, step 4799, 1 group, trunk byte-identical)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py ckpt_integrity"
  EXPECT: CKPT INTEGRITY VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=3fc29940bf97f32daa5aa7f1f715f2895f4c7c69f12c21c441cfc14a6190c0dc; exit=0; EXPECT=matched; output-sha256=2280b9b97dce2f590fc9bb564344a95d260a228b8b4bcae7a1e28f81aafe6510; output-bytes=116; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G14: no credential patterns in tracked files
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py secrets"
  EXPECT: SECRETS SCAN CLEAN
  EVIDENCE: automatic-evidence=v1; definition-sha256=380a8309f3389b0ecb62c536fd68fcb3a191ce8a9b0433ef49d854abc2189d1c; exit=0; EXPECT=matched; output-sha256=e474e9dfba6a80fb09d714926ba182e840452ed7262992d8993d0bf460658739; output-bytes=137; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G15: CI config actually runs lint+tests
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py ci"
  EXPECT: CI CONFIG VERIFIED
  EVIDENCE: automatic-evidence=v1; definition-sha256=905500db852b0156078d0f24372f08930ccca752d076ab3ea60f9c2628af0c75; exit=0; EXPECT=matched; output-sha256=4f5efacb83c08c019d7ecc848283764afa79f2908e4cdddecfa2f32dae8afa43; output-bytes=37; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G16: shipped calibration meets the plan ECE gate on the shipped ckpt (held-out synth, shipped temps)
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py temps"
  EXPECT: SHIPPED CALIBRATION MEETS ECE GATE
  EVIDENCE: automatic-evidence=v1; definition-sha256=625e92b8caa1b1fdc4abff852f5b90946a3fd88fe6e9e21ba33807621e84cad7; exit=0; EXPECT=matched; output-sha256=d2c1de060fa64d39709858ec6c719b58d3cfe34488e2a4330ba1d93a5003e069; output-bytes=263; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries

- [x] G17: shipped per-label risk thresholds reproduce exactly from the shipped ckpt
  CHECK: bash -lc "cd /Users/eulogikdeveloper/Documents/pico-type && .venv/bin/python /var/folders/6m/l_wd40y91jqbj36ty4nz2skm0000gn/T/opencode/verify_all.py thresholds"
  EXPECT: RISK THRESHOLDS REPRODUCE EXACTLY
  EVIDENCE: automatic-evidence=v1; definition-sha256=a5d0c3e6997896275a77039ca4cd52648853dee58d57d45ee30bee4df92d246e; exit=0; EXPECT=matched; output-sha256=e1fb7bc8b5c14589e6917f85a27a7b3bce88b70b0ccbf2f2bd0960cf877b4e5b; output-bytes=58; shell=/bin/sh; cwd=/Users/eulogikdeveloper/Documents/pico-type; path=afc7568fe86c/59 entries
