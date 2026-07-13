"""
blind_eval_sweep.py

Runs the full test_bench.json case set through the LIVE inference path
(generate_live_strategy / process_live_case) instead of the labeled
generate_agent_strategy path used by orchestrator.py's main sweep.

Why this exists:
The original 400-case sweep in orchestrator.py always tells the model the
correct verdict in advance ("Expected Action: {expected_verdict}...") - it
only ever tests whether the model can WRITE A JUSTIFICATION for an answer
it's already been given, never whether it can DETERMINE the right answer on
its own. This script closes that gap by running every case blind, exactly
the way a real customer's live narrative would be processed, and additionally
scores whether the model's own verdict matches the labeled ground truth
(a metric the original sweep never computes at all).

Usage:
    uv run python blind_eval_sweep.py

Requires the same .env (OPENAI_API_KEY, PINECONE_API_KEY) and test_bench.json
as orchestrator.py. Does not modify or interfere with the original sweep or
its production_audit_trail.jsonl output.
"""

import json
from datetime import datetime, timezone

from orchestrator import ChargebackOrchestrator, EVAL_JUDGE_PROMPT


def run_judge(openai_client, raw_narrative, rule_context, strategy_text, case_id):
    """
    Same RAG Triad judge call used inside orchestrator.py's process_case,
    factored out here so it can be reused against live-path output.
    """
    try:
        judge_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EVAL_JUDGE_PROMPT},
                {
                    "role": "user",
                    "content": f"""
Customer's Original Complaint:
{raw_narrative}

Retrieved PDF Rules Context:
{rule_context}

Agent's Generated Response Strategy:
{strategy_text}
"""
                }
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        eval_metrics = json.loads(judge_response.choices[0].message.content)
        return (
            float(eval_metrics.get("context_relevance_score", 0.0)),
            float(eval_metrics.get("groundedness_score", 0.0)),
            float(eval_metrics.get("answer_relevance_score", 0.0)),
        )
    except Exception as e:
        print(f"⚠️ Judge failed for case {case_id}: {e}")
        return 0.0, 0.0, 0.0


def run_blind_sweep():
    orchestrator = ChargebackOrchestrator()

    if not orchestrator.test_cases_db:
        print("⚠️ No test cases found in test_bench.json. Run sample_generator.py first.")
        return

    log_output_path = "blind_audit_trail.jsonl"
    sorted_keys = sorted(
        orchestrator.test_cases_db.keys(),
        key=lambda x: int(x) if x.isdigit() else x
    )

    total_context = 0.0
    total_groundedness = 0.0
    total_answer_relevance = 0.0
    verdict_matches = 0
    rejected_count = 0  # cases where the model refused to give a real verdict at all
    processed_count = 0

    print(f"🚀 Initiating BLIND Evaluation Sweep for {len(sorted_keys)} cases (no expected_verdict given)...")

    with open(log_output_path, "w", encoding="utf-8") as audit_log:
        for case_id in sorted_keys:
            record = orchestrator.test_cases_db[case_id]
            raw_narrative = record.get("narrative", "Default dispute narrative.")
            network_brand = record.get("network", "Visa")
            expected_verdict = str(record.get("expected_verdict", "CHALLENGE")).strip().upper()

            try:
                # This is the real blind path: no answer given in advance.
                result = orchestrator.process_live_case(raw_narrative, network_brand)
            except Exception as e:
                print(f"⚠️ process_live_case failed for case {case_id}: {e}")
                continue

            model_verdict = str(result["verdict"]).strip().upper()

            if model_verdict == "REJECTED":
                rejected_count += 1

            verdict_match = (model_verdict == expected_verdict)
            if verdict_match:
                verdict_matches += 1

            # Reconstruct the same judge-harness string format used elsewhere,
            # so groundedness/relevance scoring stays comparable to the
            # original sweep's methodology.
            strategy_text = (
                f"VERDICT: {result['verdict']} | "
                f"CITED RULE: {result['cited_rule_id']} | "
                f"RATIONALE: {result['defense_rationale']} | "
                f"PROOFS: {', '.join(result['evidentiary_requirements'])}"
            )

            context_relevance, groundedness, answer_relevance = run_judge(
                orchestrator.openai_client,
                raw_narrative,
                result["rule_context"],
                strategy_text,
                case_id
            )

            total_context += context_relevance
            total_groundedness += groundedness
            total_answer_relevance += answer_relevance
            processed_count += 1

            log_frame = {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "case_id": case_id,
                "network": network_brand,
                "expected_verdict": expected_verdict,
                "model_verdict": model_verdict,
                "verdict_match": verdict_match,
                "enriched_query": result["enriched_query"],
                "cited_rule_id": result["cited_rule_id"],
                "strategy_generated": strategy_text,
                "retrieved_context": result["rule_context"],
                "scores": {
                    "context_relevance": context_relevance,
                    "groundedness": groundedness,
                    "answer_relevance": answer_relevance,
                }
            }
            audit_log.write(json.dumps(log_frame) + "\n")

            print(f"⚡ [{case_id}] expected={expected_verdict} model={model_verdict} "
                  f"match={verdict_match} groundedness={groundedness:.2f}")

    if processed_count == 0:
        print("⚠️ No cases were successfully processed.")
        return

    avg_context = total_context / processed_count
    avg_groundedness = total_groundedness / processed_count
    avg_answer = total_answer_relevance / processed_count
    verdict_accuracy = verdict_matches / processed_count

    print("\n🏆 [BLIND SWEEP RESULTS — no answer given in advance]")
    print(f"🏁 Total Cases Evaluated: {processed_count} / {len(sorted_keys)}")
    print("──────────────────────────────────────────────────")
    print(f"🎯 Verdict Accuracy (model vs. labeled ground truth): {verdict_accuracy:.4f}")
    print(f"🚫 Cases the model REJECTED as out-of-scope (should be ~0 on real cases): {rejected_count}")
    print("──────────────────────────────────────────────────")
    print(f"🌲 Macro Average Context Relevance : {avg_context:.4f}")
    print(f"⚓ Macro Average Groundedness      : {avg_groundedness:.4f}")
    print(f"🎯 Macro Average Answer Relevance   : {avg_answer:.4f}")
    print("──────────────────────────────────────────────────")
    print(f"📝 Structured Audit File Output    : {log_output_path}")
    print("──────────────────────────────────────────────────\n")
    print(
        "Compare these numbers against production_audit_trail.jsonl (the labeled-answer "
        "sweep). A meaningful drop here - especially in verdict_accuracy or groundedness - "
        "is the true measure of system capability, since this run never told the model "
        "the correct answer in advance."
    )


if __name__ == "__main__":
    run_blind_sweep()