from pydantic import BaseModel, Field
from typing import List

class EliminatedScenario(BaseModel):
    scenario: str = Field(description="The ruled-out explanation")
    reason: str = Field(description="Why it's ruled out")




class Schema(BaseModel):

    problem: str = Field(
        description="The prompt content — can be anything from a short piece of information ,a news snippet, to a fully explained, detail-rich scenario. No fixed length or format required; "
    )
    observations: List[str] = Field(
        description="A list of observations made from the problem statement"
    )
    possible_scenario: List[str] = Field(
        description="A list of plausible explanations or hypotheses that could account for the clue in `problem`. Each item should be a distinct, concrete possibility — not a rephrasing of another item."
    )
    impossible_scenario: List[EliminatedScenario]

    deduction: str = Field(
        description="The connective reasoning in Holmes's voice: walks from `problem`, through why the ruled_out options in `analysing_possibilities` fail, to why the surviving option(s) fit. Explain the 'why', don't just restate the options."
    )
    final_answer: str = Field(
        description="The concise final verdict reached at the end of `deduction` — one to two sentences stating the conclusion directly, with no new reasoning introduced here."
    )

