# Candy Crush Contract and Segmentation Ablation Results

Run root:

```
runs/candy_crush_contract_segmentation_ablation/
```

## Run Inventory

| Variant | Seed | Run dir | Commit | Status | Notes |
| --- | ---: | --- | --- | --- | --- |
| full | 0 | TBD | TBD | TBD |  |
| full | 1 | TBD | TBD | TBD |  |
| full | 2 | TBD | TBD | TBD |  |
| w/o effect contract | 0 | TBD | TBD | TBD |  |
| w/o effect contract | 1 | TBD | TBD | TBD |  |
| w/o effect contract | 2 | TBD | TBD | TBD |  |
| raw delta contract | 0 | TBD | TBD | TBD |  |
| raw delta contract | 1 | TBD | TBD | TBD |  |
| raw delta contract | 2 | TBD | TBD | TBD |  |
| heuristic-only segmentation | 0 | TBD | TBD | TBD |  |
| heuristic-only segmentation | 1 | TBD | TBD | TBD |  |
| heuristic-only segmentation | 2 | TBD | TBD | TBD |  |

## Main Table

| Variant | Mean reward | AURC | Active skills | Skill reuse | Admit rate | False admit | Contract pass | Segments / ep | Mean seg len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| w/o effect contract | TBD | TBD | TBD | TBD | TBD | TBD | n/a | TBD | TBD |
| raw delta contract | TBD | TBD | TBD | TBD | TBD | TBD | n/a | TBD | TBD |
| heuristic-only segmentation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Interpretation Prompts

### Is the effect contract necessary?

Compare `full` against `w/o effect contract`.

- Reward gap:
- Skill reuse gap:
- False-admit / verification gap:
- Conclusion:

### Is the contract only predicate counting?

Compare `full` against `raw delta contract`.

- Reward gap:
- Contract stability:
- False-admit / noisy-skill evidence:
- Conclusion:

### Is learned segmentation a core contribution?

Compare `full` against `heuristic-only segmentation`.

- Reward gap:
- Segment granularity gap:
- Active skill quality / reuse gap:
- Difference from weak SkillRL setting:
- Conclusion:

