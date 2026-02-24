# llava_scripts Legacy Migration

## What changed
- Historical scripts under `scripts/llava_scripts/` have been moved to `legacy/llava_scripts/`.
- A compatibility wrapper is kept at each original path under `scripts/llava_scripts/`.
- Wrappers print a deprecation warning and forward all arguments to the matching script in `legacy/llava_scripts/`.

## Why
- Keep GEM official training/evaluation entrypoints clean and maintainable.
- Isolate upstream LLaVA/historical utilities from the supported GEM pipeline.

## Compatibility policy
- Compatibility wrappers are guaranteed for one release cycle.
- Wrappers will be removed in the next major release.

## Migration guidance
1. Replace any direct call from:
   - `scripts/llava_scripts/...`
2. To:
   - `legacy/llava_scripts/...`
3. Prefer the new unified pipeline for GEM workflows:
   - `python scripts/gem_pipeline.py <subcommand> --config configs/pipelines/gem_default.yaml`

## Example
```bash
# old (still works during compatibility window)
bash scripts/llava_scripts/finetune.sh

# new
bash legacy/llava_scripts/finetune.sh
```
