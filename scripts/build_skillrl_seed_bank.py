#!/usr/bin/env python3
"""Build SkillRL-style seed skill banks for ALFWorld and WebShop.

The previous seed bank (gpt54_textenv_skillbank) contained 4 generic
skills (EXECUTE/EXPLORE/...) whose descriptions restated effects and
carried no task semantics.  This script replaces it with a hand-crafted
bank whose skills:

  * use the task-semantic phase:TAG identities that the new phase
    detectors emit (search:EXPLORE, acquire:COLLECT, ...), so extracted
    segments merge into these seeds instead of spawning junk skills;
  * carry *prescriptive* descriptions in the SkillRL experience style
    (WHEN to use / WHAT to do / what to AVOID), distilled from the
    SkillRL-SFT-Data principles that reach >90% on ALFWorld;
  * expose machine-checkable predicates built on the new transferable
    facts (holding=, obj_state=, location_type=, target_visible=).

Output: labeling/output/skillrl_seed_bank/{alfworld,webshop}/skill_bank.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from skill_agents_grpo.skill_bank.bank import SkillBankMVP
from skill_agents_grpo.stage3_mvp.schemas import (
    ExecutionHint,
    Protocol,
    Skill,
    SkillEffectsContract,
)

OUT_DIR = PROJECT_ROOT / "labeling" / "output" / "skillrl_seed_bank"


def _mk(
    skill_id: str,
    name: str,
    description: str,
    preconditions: list,
    steps: list,
    success: list,
    abort: list,
    pred_success: list,
    eff_add: set,
    eff_del: set,
    hint: str,
    failure_modes: list,
) -> Skill:
    contract = SkillEffectsContract(
        skill_id=skill_id,
        name=name,
        description=description,
        eff_add=set(eff_add),
        eff_del=set(eff_del),
        support={p: 5 for p in (set(eff_add) | set(eff_del))},
        n_instances=5,
    )
    protocol = Protocol(
        preconditions=preconditions,
        steps=steps,
        success_criteria=success,
        abort_criteria=abort,
        predicate_success=pred_success,
        expected_duration=8,
    )
    return Skill(
        skill_id=skill_id,
        name=name,
        strategic_description=description,
        protocol=protocol,
        contract=contract,
        execution_hint=ExecutionHint(
            execution_description=hint,
            common_failure_modes=failure_modes,
            n_source_segments=5,
        ),
        n_instances=5,
    )


# ═════════════════════════════════════════════════════════════════════
# ALFWorld — distilled from SkillRL General Principles + per-task skills
# ═════════════════════════════════════════════════════════════════════

ALFWORLD_SKILLS = [
    _mk(
        "search:EXPLORE",
        "systematic_one_pass_search",
        "When the goal object is not visible yet, search each plausible "
        "receptacle exactly once: go to it, open it if closed, and inspect "
        "the contents. Use location priors (food in fridge/countertop/"
        "sinkbasin, small items in drawers/desks/shelves, clothes in "
        "dressers). Avoid revisiting locations you already searched and "
        "never rule out a closed container without opening it.",
        preconditions=[
            "The exact object type named in the task is not visible",
            "You are not carrying the goal object",
        ],
        steps=[
            "List plausible locations for the goal object type (use priors)",
            "go to the nearest unsearched location",
            "open it if it is closed",
            "Scan the observation for the exact goal object type",
            "If absent, mark this location searched and move to the next",
        ],
        success=["The exact goal object type is visible in the observation"],
        abort=["All plausible locations have been searched once"],
        pred_success=["target_visible=*"],
        eff_add={"world.target_visible=*"},
        eff_del=set(),
        hint="Search each plausible container once, open closed ones, "
        "never revisit searched spots.",
        failure_modes=[
            "Revisiting the same already-searched location in a loop",
            "Skipping a closed container without opening it",
        ],
    ),
    _mk(
        "acquire:COLLECT",
        "grab_when_seen",
        "As soon as the exact object type named in the task is visible and "
        "reachable, take it immediately before doing anything else. Match "
        "the type exactly (a mug is not a cup; a tomato is not a potato). "
        "Never navigate to appliances or the destination receptacle before "
        "you are holding the required object.",
        preconditions=[
            "The exact goal object type is visible at your current location",
            "You are not already carrying it",
        ],
        steps=[
            "Confirm the visible object matches the task noun exactly",
            "take <object> from <receptacle>",
            "Confirm the pickup succeeded in the next observation",
        ],
        success=["You are carrying the exact goal object type"],
        abort=["The object cannot be taken after one retry"],
        pred_success=["holding=*"],
        eff_add={"world.holding=*"},
        eff_del={"world.target_visible=*"},
        hint="Take the goal object the moment you see it; exact type match only.",
        failure_modes=[
            "Walking away from a visible goal object to keep exploring",
            "Picking up a look-alike of the wrong type",
        ],
    ),
    _mk(
        "transform:EXECUTE",
        "transform_before_transport",
        "When the task requires a hot/cool/clean object and you are holding "
        "it, go straight to the matching appliance (microwave to heat, "
        "fridge to cool, sinkbasin to clean) and apply the state change "
        "BEFORE heading to the destination. Sequence: go to appliance, "
        "open it if closed, use 'heat/cool/clean <object> with <appliance>', "
        "and verify the state changed. Never visit the appliance before "
        "holding the object.",
        preconditions=[
            "The task mentions hot/heated, cool/cold, or clean",
            "You are carrying the goal object",
            "The object state change has not happened yet",
        ],
        steps=[
            "go to the matching appliance (microwave/fridge/sinkbasin)",
            "open the appliance if it is closed",
            "heat/cool/clean <object> with <appliance>",
            "Verify the observation confirms the state change",
        ],
        success=["Observation confirms the object was heated/cooled/cleaned"],
        abort=["Appliance is missing or interaction fails twice"],
        pred_success=["obj_state=*"],
        eff_add={"world.obj_state=*"},
        eff_del=set(),
        hint="Holding the object, use microwave=heat / fridge=cool / "
        "sinkbasin=clean before delivering.",
        failure_modes=[
            "Going to the destination before applying the state change",
            "Interacting with the appliance while not holding the object",
        ],
    ),
    _mk(
        "deliver:POSITION",
        "direct_post_acquire_delivery",
        "Once you hold the goal object (and any required hot/cool/clean "
        "transform is done), navigate directly to the destination "
        "receptacle named in the task, open it if closed, and 'put "
        "<object> in/on <receptacle>'. Do not explore or take detours "
        "while holding a ready-to-deliver object.",
        preconditions=[
            "You are carrying the goal object",
            "Any required state change is already applied",
        ],
        steps=[
            "go to the destination receptacle named in the task",
            "open it if it is closed",
            "put <object> in/on <receptacle>",
            "Confirm placement in the next observation",
        ],
        success=["The object is placed in/on the destination receptacle"],
        abort=["The destination cannot be found after checking the task text"],
        pred_success=["action_result=placed"],
        eff_add={"world.action_result=placed"},
        eff_del={"world.holding=*"},
        hint="Holding a ready object: go straight to the goal receptacle and place it.",
        failure_modes=[
            "Wandering to unrelated locations while holding the goal object",
            "Forgetting to open a closed destination (e.g. safe, cabinet)",
        ],
    ),
    _mk(
        "deliver:EXECUTE",
        "examine_under_light",
        "For 'look at/examine <object> under the desklamp' tasks: FIRST take "
        "the target object, THEN go to where a desklamp is (desks and "
        "sidetables), and 'use desklamp <N>' exactly once to switch it on. "
        "Do not toggle the lamp repeatedly and do not go to the lamp before "
        "holding the object.",
        preconditions=[
            "The task asks to look at or examine an object under a lamp",
            "You are carrying the target object",
        ],
        steps=[
            "go to a desk or sidetable that has a desklamp",
            "use desklamp <N>",
            "Stop — the task completes when the lamp is on while holding the object",
        ],
        success=["The desklamp is on while you hold the target object"],
        abort=["No desklamp found on any desk/sidetable/shelf"],
        pred_success=["action_result=toggled"],
        eff_add={"world.action_result=toggled"},
        eff_del=set(),
        hint="Take target first, then a single 'use desklamp N' — never toggle twice.",
        failure_modes=[
            "Toggling the lamp on and off repeatedly",
            "Activating the lamp before acquiring the target object",
        ],
    ),
    _mk(
        "search:NAVIGATE",
        "track_counts_for_multiples",
        "For 'put two <object>s' tasks: after delivering the first object, "
        "resume the search for the second one, skipping every location you "
        "already searched. Keep an internal count of how many objects are "
        "still needed and stop only when the count reaches zero.",
        preconditions=[
            "The task requires two of the same object type",
            "The first object has already been delivered",
        ],
        steps=[
            "Recall which locations were already searched",
            "go to the nearest unsearched plausible location",
            "Take the second object when visible",
            "Deliver it to the same destination receptacle",
        ],
        success=["Both required objects are placed at the destination"],
        abort=["All locations searched and no second object found"],
        pred_success=["action_result=placed"],
        eff_add={"world.action_result=placed"},
        eff_del=set(),
        hint="Two-object tasks: deliver first, then search only unsearched spots for the second.",
        failure_modes=[
            "Losing count and searching after both objects are delivered",
            "Re-searching locations already visited for the first object",
        ],
    ),
]


# ═════════════════════════════════════════════════════════════════════
# WebShop
# ═════════════════════════════════════════════════════════════════════

WEBSHOP_SKILLS = [
    _mk(
        "query:EXPLORE",
        "compose_specific_query",
        "On the search page, issue ONE search containing the key product "
        "attributes from the instruction (product type, size/volume, "
        "color/flavor, key constraint like 'sulfate free'), but leave the "
        "price out of the query. Keep it under ~8 keywords. Avoid vague "
        "one-word queries and avoid re-searching the same words twice.",
        preconditions=["You are on the search page (only action is search[...])"],
        steps=[
            "Extract product type + the 2-4 most specific attributes from the task",
            "search[<type> <attributes>]",
            "Wait for the results list",
        ],
        success=["A results page with relevant products is shown"],
        abort=["Two consecutive searches return no relevant results"],
        pred_success=[],
        eff_add={"world.page=results"},
        eff_del={"world.page=search"},
        hint="One precise search: type + size + color/flavor + constraint, no price words.",
        failure_modes=[
            "Vague single-word queries returning irrelevant items",
            "Repeating the identical query instead of refining it",
        ],
    ),
    _mk(
        "browse:NAVIGATE",
        "scan_results_against_constraints",
        "On the results page, compare each item title and price against ALL "
        "requirements including the price cap, then click the product code "
        "of the best match. If nothing on the first page satisfies the "
        "requirements, refine the search query rather than paging through "
        "many result pages.",
        preconditions=["A results list with product codes and prices is shown"],
        steps=[
            "Check each title for required attributes and each price against the budget",
            "click[<product code>] of the best-matching item",
            "If no match, click[Back to Search] and refine the query",
        ],
        success=["A product page satisfying the requirements is open"],
        abort=["Three pages browsed with no match — refine the query instead"],
        pred_success=[],
        eff_add={"world.page=product"},
        eff_del={"world.page=results"},
        hint="Match title AND price to every requirement before clicking a product.",
        failure_modes=[
            "Clicking an over-budget item because only the title was checked",
            "Endless paging instead of refining the search",
        ],
    ),
    _mk(
        "verify:EXECUTE",
        "select_options_before_buying",
        "On the product page, click every option the instruction requires "
        "(size, color, flavor, count) so it is selected, and confirm the "
        "displayed price is within budget. Only proceed to buy after all "
        "required options are selected. If an option or the price does not "
        "match, go back to the results instead of buying a wrong item.",
        preconditions=["You are on a product page with clickable options"],
        steps=[
            "List the options the instruction requires",
            "click[<option>] for each required option",
            "Confirm the price still fits the budget",
        ],
        success=["All required options selected and price within budget"],
        abort=["A required option does not exist on this product"],
        pred_success=[],
        eff_add={"world.options=selected"},
        eff_del=set(),
        hint="Click every required option (size/color/count) before buying; recheck price.",
        failure_modes=[
            "Buying with default options when the task specified a size/color",
            "Ignoring a price increase after option selection",
        ],
    ),
    _mk(
        "purchase:EXECUTE",
        "buy_once_verified",
        "Click 'Buy Now' exactly once, and only after the product matches "
        "the type, all required options are selected, and the price is "
        "within the budget. Do not buy a partially-matching product just to "
        "finish the episode early.",
        preconditions=[
            "The product matches all requirements",
            "All required options are selected",
        ],
        steps=["click[Buy Now]"],
        success=["The purchase is confirmed"],
        abort=["Any requirement is unmet — go back instead of buying"],
        pred_success=[],
        eff_add={"world.purchased=true"},
        eff_del=set(),
        hint="Buy only after type+options+price all match; never buy to bail out.",
        failure_modes=[
            "Buying an item missing a required attribute to end the episode",
        ],
    ),
]


def main() -> None:
    for game, skills in (("alfworld", ALFWORLD_SKILLS), ("webshop", WEBSHOP_SKILLS)):
        bank_path = OUT_DIR / game / "skill_bank.jsonl"
        bank = SkillBankMVP(str(bank_path))
        for sk in skills:
            bank.add_or_update_skill(sk)
        bank.save()
        print(f"{game}: wrote {len(skills)} skills -> {bank_path}")


if __name__ == "__main__":
    main()
