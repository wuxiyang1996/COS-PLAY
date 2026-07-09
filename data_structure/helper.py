# This code is used to store the assistive functions for the data structure function which canot be included 
# in the data structure class due to some constraints.

# Note: Experience import removed to avoid circular import with experience.py
# If needed, import Experience using TYPE_CHECKING or import inside functions
from API_func import ask_model
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from data_structure.experience import Experience

# # This is a helpere function to generate the intentions for the experience.
# # Only for expereince labeling process. We should have intentions for our own agents.
# def generate_intentions(curr_experience: Experience,  history: List[Experience]):
#     prev_summary_list = [experience.summary for experience in history]
#     curr_state = curr_experience.state
#     curr_action = curr_experience.action
#     curr_next_state = curr_experience.next_state
#     curr_sub_task = curr_experience.sub_task
#     prompt = (
#         f"Generate the intentions for the experience, while the intentions marks the motivation for agent to take the action and achieve the sub-task."
#         f"Also, the historical state over the last few steps should be considered."
#         f"The current state is {curr_state}, the current action is {curr_action}, the current next state is {curr_next_state}, the current sub-task is {curr_sub_task}. "
#         f"The history is {prev_summary_list}. Given in the time-wise order."
#     )
#     intentions = ask_model(prompt)
#     return intentions

# We may need some functions to convert teh experience from other sources into the experience data structure.