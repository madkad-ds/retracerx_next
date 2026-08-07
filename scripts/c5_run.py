#!/usr/bin/env python3
"""c5_run.py — inject the real Phase C rankers into the FROZEN D4 harness and run
its own main(). Leaves d4_harness.py untouched (Phase C consumes Phase D, not the
reverse). Same CLI as the harness.

    python scripts/c5_run.py --ranker grounded_agent --splits disease_cold --candidate-mode plus_true
    python scripts/c5_run.py --ranker ungrounded_llm --splits random disease_cold --candidate-mode plus_true
    python scripts/c5_run.py --ranker kge  --splits random        # still works (refactor check)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_harness as H                      # frozen Phase D artifact — not edited
from c5_rankers import GroundedAgentRanker, UngroundedLLMRanker, GraphRAGRanker

# mutate the SAME dict object main() reads (choices + lookup both see this)
H.RANKERS["grounded_agent"] = GroundedAgentRanker
H.RANKERS["ungrounded_llm"] = UngroundedLLMRanker
H.RANKERS["graph_rag"] = GraphRAGRanker

if __name__ == "__main__":
    H.main()
