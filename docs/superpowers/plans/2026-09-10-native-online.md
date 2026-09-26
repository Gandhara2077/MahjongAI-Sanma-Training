# Native online reuse implementation plan

Goal: reuse the existing v4 Rust encoder and MortalBatchAgent safely for sanma online play.
Design approved in conversation: filter online actions using native rules, preserve offline v4 ABI, keep compat fallback and production run untouched.

- [x] Add failing tests on the real MortalBatchAgent scene queue: frozen-wait North is masked, legal North is not bypassed by quick evaluation.
- [x] Filter only the online sanma mask with native can_nukidora; exclude legal North from forced-discard shortcut. Do not change PlayerState.encode_obs or GameplayLoader.
- [x] Run Rust sanma regression tests and compile an isolated candidate without copying over either installed pyd.
- [x] Run native arena smoke using the isolated binary; keep quick_eval disabled for comparison with compat.
- [x] Measure frozen-model native/compat throughput in an uncontended GPU window before choosing deployment. No production switch or automatic promotion in this stage. See reports/native-online-20260916.md for real-model parity and bounded online resume evidence.

Files: Mortal/libriichi/src/agent/mortal.rs (implementation and colocated Rust tests); checks/native_online_smoke.py (isolated ABI/arena smoke); reports/native-online-20260910.md (evidence).
Validation: cargo test --locked -p libriichi --features sanma --lib; isolated release build; smoke with explicit extension path; git diff --check.
Review: controller review approved previously as fallback; do not represent it as independent review. No commit or push.
