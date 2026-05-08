# CLAUDE.md - qm_anderson

Production-grade directives for this physics research codebase. These override
default assistant behavior. The governing loop: **gather context → take action
→ verify → repeat.**

---

## 1. Scientific Honesty is Non-Negotiable

### Never assert what has not been verified
Before citing a paper for any claim:
- Confirm the paper actually makes that claim (re-read or re-fetch if needed).
- Example of past failure: y=1.12 was attributed to Slevin & Ohtsuki 1999 PRL 82,382 -
  but that paper fits y FREELY and obtains |y|≈2.8, not 1.12.

### Never state a physical result without a citation or direct measurement
Either:
- Quote the literature (with verified author, year, journal, value), or
- Quote our data (with file path, parameter settings, exact output).

Phrases like "it is known that..." or "typically..." without a source are forbidden.

### Work from raw data
When the user shows output (numbers, plots, error messages): analyze exactly what
the output says before proposing any fix. Never guess a cause without evidence in
the data. If you need more diagnostic information, say so explicitly.

---

## 2. Pre-Work

### Read before you edit
Before editing ANY file, re-read the relevant section. After long conversations,
context compaction may have discarded what you previously read. Use Read tool with
offset+limit for files > 200 lines. After editing, read the changed section again
to confirm the edit applied correctly.

### Separate plan from execution
When the user asks to "think about this" or "propose changes": output only the
analysis and proposal. No code until the user approves. If you spot a problem in
an approved plan, flag it before implementing - do not improvise silently.

### Explicit phases for multi-file changes
Never change more than 5 files per response. State which files will be changed and
why. For changes to the physics (E value, FSS parameters, p-grid) that affect runs:
propose first, wait for confirmation, then implement.

---

## 3. Code Quality

### Verification is mandatory
After code changes, run the test suite:
```bash
pixi run test
```
Do NOT report a change as complete without running tests. If tests fail, fix before
reporting. If no test exists for the changed code, say so explicitly.

### No unverified citations in code or comments
If a comment says "Slevin & Ohtsuki 1999" or "Rodriguez 2011" - that paper must
actually contain the cited claim. The two most common errors in this project:
- Attributing y=1.12 to S&O99 (their value is |y|≈2.8; our 1.12 is empirical)
- Attributing the δ window scan to Rodriguez2011 (their test is polynomial-order
  perturbation, not window-width scanning)

### Write human code
No robotic comment blocks. No "# Step 1: initialize" style headers. If three
experienced physicists would all write it the same way, that's the way.

---

## 4. Physics-Specific Rules

### Energy choice is a physics decision, not a parameter choice
Before changing E in sc_dilution.py or bcc_dilution.py, state:
1. The ξ_cross argument (BDI-AI crossover scale vs. M_min)
2. The expected |b| relative to benchmark (Anderson, b≈0.1)
3. The literature value of p_c^Q at that E (Ujfalusi 2014 if available)

### FSS model changes require benchmark regression
Any change to fit_fss parameters (poly_order, n_corrections, y_irrel) must be
validated against the Anderson benchmark:
```bash
pixi run python runs/sc_benchmark.py --refit
```
W_c must stay within 0.30 of 16.53; ν within 0.10 of 1.571. If not: revert.

### χ²/dof > 5.0 after window scan = stop and diagnose
Do NOT manually force a window to get acceptable χ². A forced window masks the
real problem. Acceptable diagnostic actions:
- Print the Λ_M table (look for crossing structure)
- Print crossing values (spurious or real?)
- Check E vs. half-bandwidth ratio
- Check b sign and magnitude

### Crossing detector is the primary estimator
Use pairwise sign-change detector (not minimum spread) to find p_q_rough.
Minimum spread fails when corrections-to-scaling stagger the crossings.

---

## 5. Session Continuity

### Context decay awareness
After 10+ messages, re-read any file before editing. Auto-compaction silently
discards context. Always check the actual file state, never trust memory.

### RUNBOOK and docs are the authoritative state
The source of truth for run status, failure modes, and methodological decisions
is RUNBOOK.md. The source of truth for methodology is docs/methodology.md.
Memory files in ~/.claude/... are secondary. If they conflict, RUNBOOK wins.

### Memory files: update when something non-obvious is learned
Non-obvious = something that future-me would not derive from the code. Examples:
- y=1.12 is empirical, not from S&O99
- Rodriguez2011 stability test is polynomial-order, not window-width
- E=0.1 fails for SC because ξ_cross ≈ M_max (not just "small corrections")

---

## 6. File System as State

Write intermediate results to data files. Use checkpoint files. Do not hold
large numerical arrays in context - read them from HDF5 when needed.

For greengoblin runs:
- Always confirm checkpoint names match OUTFILE names (E3p1 suffix if E=3.1)
- Never delete a checkpoint without confirming no partial run is using it

---

## 7. Error Recovery

### Two failed attempts = stop and rethink
If a fix does not work after two iterations: pause. Re-read the relevant section
of RUNBOOK and docs/methodology.md from scratch. Explain where the mental model
was wrong. Propose something structurally different.

### Failure mode taxonomy is in RUNBOOK §Step 3
When an FSS result is unexpected, check against documented failure modes A–F
before proposing a new fix. Document any new failure mode in RUNBOOK immediately.

---

## 8. Verified Reference Values

These are verified from papers or our own data. Cite these, not memory:

| Quantity | Value | Source (verified) |
|----------|-------|-------------------|
| W_c(3D-AI, box) | 16.54 | S&O 1999 PRL 82,382 Table II |
| ν(3D-AI) | 1.57–1.58 (S&O99); 1.590 (Rodriguez11) | Both verified |
| |y| for Λ observable | 2.8–3.9 | S&O99 Table II (fitted freely) |
| |y| for fixed-λ FSS | 1.7 | Rodriguez11 Table V |
| p_c^Q(SC, E=3.1) | 0.50628 (0.50606..0.50647) | Ujfalusi 2014 PRB 90,174203 Table II |
| p_c^Q(SC, E=0.7) | 0.38402 (0.38387..0.38418) | Ujfalusi 2014 PRB 90,174203 Table II |
| p_c^Q(SC, E=0.1) | 0.45384 (0.45365..0.45402) | Ujfalusi 2014 PRB 90,174203 Table II |
| p_c^Q(SC, E=0) | → 1 | Ujfalusi 2014 (MFSS convergence failure near E=0) |
| p_c(SC) classical | 0.3116 | Standard site percolation MC |
| p_c(BCC) classical | 0.2457 | Standard site percolation MC |
| Our SC benchmark W_c | 16.6174 ± 0.016 | data/sc_benchmark.h5 |
| Our SC benchmark ν | 1.6205 ± 0.009 | data/sc_benchmark.h5 |
| Our SC E=0 result | p_q=0.591, ν=2.0, b≈0 | data/sc_dilution_E0.h5 |
| Our SC E=0.1 result | p_q=0.372 (spurious), b=−0.153, χ²=1543 | data/sc_dilution_E0p1.h5 |
| Our SC E=0.7 result | crossings at 0.316/0.318/0.319 (Failure F: ξ_perc≈20>>M) | data/sc_dilution_E0p7.h5 |
| Our SC E=3.1, M=[4,6,8,10] | single crossing 0.482, χ²/dof=213, b=−2.208 (Failure G) | data/mc_dilution_E3p1.h5 |
| ξ_perc(p_q, SC, E=3.1) | 4.2 (= (0.195)^{−0.876}); NOT 7 as previously stated | measured/derived |
| p*(M=4) SC E=3.1 | 0.517 > p_q → M=4 useless for FSS | derived: p_c + 4^{−1.142} |
| **Citation error fixed** | PRB 90,**174203** (not 245136) | Ujfalusi & Varga 2014 |
