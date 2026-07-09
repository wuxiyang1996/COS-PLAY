# F2 — paper consistency sweep

Run these checks against the paper LaTeX before submitting.  All can
be done with `rg` from the paper root.

## Architectural vocabulary

The paper must consistently treat the **Harness** as a *runtime*
validation layer, **not** as a separate agent.  Sweep:

```bash
rg -n -i 'harness agent|validator agent|harness module' paper/
```

For every hit, rewrite as one of:

* "the harness" (when referring to the runtime layer)
* "`SkillHarness`" (when referring to the class)
* "the runtime validation layer (harness)" (introductory mentions)

## Skill Bank Agent vs. Skill Agent

The offline pipeline is the **Skill Bank Agent** (capital B).
"Skill agent" is ambiguous — sweep:

```bash
rg -n 'Skill agent|skill agent\b' paper/
```

Rewrite as:

* "Skill Bank Agent" — for the full offline pipeline
  (extractor + crafter + promotion gate).
* "the Skill Bank `B_n`" — for the persistent data structure alone.

## Decision Agent

The actor is the **Decision Agent**.  Sweep:

```bash
rg -n -i 'actor agent|policy agent|action agent' paper/
```

Rewrite as **Decision Agent**.

## LoRA naming consistency

Three named LoRAs only:

* `action_taking` — low-level action emission
* `skill_selection` — skill ranking / selection
* `intention`     — intention update

Sweep:

```bash
rg -n -i 'skill[_ -]?picker|skill[_ -]?chooser|action[_ -]?lora|intent[_ -]?lora' paper/
```

## Spelling

```bash
rg -ni 'harness skill|skil bank|coevoluton|skil bridge' paper/
```

These are the most common typos seen in the working drafts.

## Numerical placeholders

Every table cell must be either a number or `-`.  Look for placeholders:

```bash
rg -n 'TODO|TBD|XXX|\?\?\?' paper/
```

## Cross-references

Each generator script in the stubs section produces a JSON summary +
a PNG.  When promoting the stub into the LaTeX paper, reference the
exact JSON (so the generator command is reproducible from the paper)
in the figure caption.
