# Code Review Report — 2026-04-02

Three review agents (reuse, quality, efficiency) analyzed the full codebase.

---

## Actionable Findings

### 1. Float/double parsing uses unnecessary byte roundtrip (parser.rs)

**7 locations** use `f32::from_le_bytes(cursor.read_fixed32()?.to_le_bytes())` instead of the idiomatic `f32::from_bits(cursor.read_fixed32()?)`. Same for f64/fixed64.

`read_fixed32()` returns `u32`. Calling `.to_le_bytes()` then `from_le_bytes()` is a no-op identity — the bits don't change. `from_bits()` does the same thing in one step.

**Locations in `parse_tensor()`:**
- Line 240-242: `float_data` packed branch
- Line 246-248: `float_data` single branch
- Line 276-278: `double_data` packed branch
- Line 282-284: `double_data` single branch

**Locations in `parse_attribute()`:**
- Line 364: `attr.f` single float
- Line 380-382: `attr.floats` packed branch
- Line 386-388: `attr.floats` single branch

**Fix:** Replace `f32::from_le_bytes(x.read_fixed32()?.to_le_bytes())` with `f32::from_bits(x.read_fixed32()?)` and `f64::from_le_bytes(x.read_fixed64()?.to_le_bytes())` with `f64::from_bits(x.read_fixed64()?)`.

---

### 2. Encoder allocates many temporary Vecs (encoder.rs)

Every `encode_*` function allocates its own `Vec<u8>`, then the caller copies it via `write_message_field` → `buf.extend_from_slice(&inner)`. For deeply nested models this creates hundreds of temporary allocations.

Additionally, packed field encoding (9 locations) each create a temporary `packed` Vec, write elements, then copy to `buf`.

**Packed field locations in `encode_tensor()`:** lines 209, 226, 236, 250, 263, 273
**In `encode_sparse_tensor()`:** line 312
**In `encode_attribute()`:** lines 350, 360

**Potential fix:** Refactor `encode_*` to accept `&mut Vec<u8>` instead of returning `Vec<u8>`. For packed fields, could reserve space for the length prefix and backfill. This is a larger refactor — assess if encoding performance matters for your use case first.

---

## Confirmed Correct (No Action Needed)

### write_string_field vs write_string_field_always (encoder.rs, lines 37-50)

The asymmetry is correct per protobuf semantics:
- `write_string_field` skips empty strings — correct for optional scalar fields (name, domain, doc_string)
- `write_string_field_always` preserves empty strings — correct for repeated fields where position matters (node inputs/outputs)

### OpType enum consistency (ast.rs)

All 236 variants are present in the enum definition, `From<&str>` impl, and `op_type_to_str()`. The `From<&str>` impl includes two aliases (`"Cumsum"|"CumSum"`, `"Hardmax"|"HardMax"`) for compat. No inconsistencies found.

### String allocations in parser (parser.rs)

Every `cursor.read_string()?.to_string()` allocates from a borrowed `&str`. This is unavoidable since AST structs own their strings.

---

## Deferred / Low Priority

- **Packed field parsing pattern** (parser.rs): The `match wire_type { LengthDelimited => sub-cursor loop, _ => single }` pattern repeats ~15 times. Could extract a helper, but the current code is clear and the abstraction would add complexity for marginal benefit.
- **OpType triple-match duplication** (ast.rs): ~190 match arms repeated across enum def, `From<&str>`, and `op_type_to_str`. Could use a macro or lookup table, but explicit matches are easy to maintain and verify.
- **Varint overflow checks** (wire.rs, lines 48-60): Correct implementation, negligible overhead.
