from llm import get_llm
from schemas import Schema
import json
import os

contexts = ['crime/mystery scenario', 'everyday logical reasoning', 'scientific reasoning', 'pure information/facts']
llm = get_llm(output_schema=Schema)

OUTPUT_PATH = "D:/data/dataset.jsonl"
TOTAL_LOOPS = 50
EXAMPLES_PER_LOOP = len(contexts) * 5   # 4 contexts * 5 difficulties = 20
TOTAL_TARGET = TOTAL_LOOPS * EXAMPLES_PER_LOOP

def get_line_count(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

existing_count = get_line_count(OUTPUT_PATH)
start_loop = existing_count // EXAMPLES_PER_LOOP
print(f"Found {existing_count} existing examples. Resuming from loop {start_loop}/{TOTAL_LOOPS}")
print(f"Target: {TOTAL_TARGET} total examples")

success_count = 0
fail_count = 0

with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
    for loop in range(start_loop,TOTAL_LOOPS):
        for context in contexts:
            for difficulty in range(1,6):
                prompt = f"""Generate one reasoning-style dataset sample.

                Domain: {context}
                Target difficulty: {difficulty} (1=single obvious inference, 5=multiple interacting clues with several plausible-looking traps)

                Rules:
                    - Every fact used in `observations` MUST come from from a fact already stated in `problem` or known facts, don't include any assumptions.
                    - Every fact used in `observations`, `impossible_scenario` reasons, and `deduction` MUST come from `problem` or from a fact already stated in `observations`.
                    - Do NOT assert the absence, non-occurrence, or non-existence of something (e.g. "no alarm was triggered," "no delivery was scheduled," "records show nothing," "no footprints were found") unless a prior `observations` entry explicitly states that the relevant check, log review, or witness canvass was performed. If an absence is needed for the deduction, first add an observation stating the check happened, before using its result.
                    - Do not introduce new named entities, events, dates, or documents that are not present in `problem`.
                """
                try:
                    result = llm.invoke(prompt)
                    record = result.model_dump()   # pydantic model -> dict
                    record["context"] = context
                    record["difficulty"] = difficulty
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()  # write immediately so you don't lose progress on a crash
                    success_count += 1

                except Exception as e:
                    fail_count += 1
                    print(f"[loop {loop}] [{context}] FAILED: {e}")
                    continue  # skip this sample, move to the next
                total_written = existing_count + success_count
                remaining = TOTAL_TARGET - total_written
                print(f"Generated: {total_written}/{TOTAL_TARGET} | Remaining: {remaining} | fails: {fail_count}")

print(f"Finished. success={success_count} fail={fail_count}")


# print(result.problem)
# print(result.observations)
# print(result.possible_scenario)
# print(result.impossible_scenario)
# print(result.deduction)
# print(result.final_answer)
# print(result.difficulty)
