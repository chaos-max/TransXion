"""Build prompts for LLM agent."""

import json
from typing import Dict, Any


def build_prompt(state_json: Dict[str, Any]) -> str:
    """Build prompt for LLM agent.

    Args:
        state_json: State including candidates, budget, allowed_tools, etc.

    Returns:
        Prompt string
    """
    cluster_id = state_json["cluster_id"]
    step_idx = state_json["step_idx"]
    cluster_summary = state_json["cluster_summary"]
    candidates = state_json["candidates"]
    budget_left = state_json["budget_left"]
    allowed_tools = state_json["allowed_tools"]
    allowed_params = state_json["allowed_params"]
    history = state_json.get("history", {})
    last_failure = state_json.get("last_failure")

    # Build candidates sections in the order of allowed_tools
    candidates_section = ""

    # Define candidate builders for each tool
    def build_inject_section():
        inject_cands = candidates.get("inject_candidates", [])
        if len(inject_cands) == 0:
            return ""
        lines = [
            "### Inject Candidates (Top edges)",
            "You MUST select edge_id from this list (use the EXACT edge_id value, NOT the number):"
        ]
        for i, cand in enumerate(inject_cands[:10]):
            lines.append(
                f"[{i+1}] edge_id='{cand['edge_id']}' (from={cand['from']}, to={cand['to']}, "
                f"currency={cand['payment_currency']}, amount={cand['amount_paid']}, "
                f"score_bridge={cand['score_bridge']})"
            )
        return "\n".join(lines) + "\n"

    def build_merge_section():
        merge_cands = candidates.get("merge_candidates", [])
        if len(merge_cands) == 0:
            return ""
        lines = [
            "### Merge Candidates (Top pairs)",
            "You MUST select pairs from this list (use the EXACT a and b values, NOT the number):"
        ]
        for i, cand in enumerate(merge_cands[:10]):
            lines.append(
                f"[{i+1}] a='{cand['a']}', b='{cand['b']}' (bank_a={cand['bank_id_a']}, bank_b={cand['bank_id_b']}, "
                f"score_jaccard={cand['score_jaccard']:.3f})"
            )
        return "\n".join(lines) + "\n"

    def build_split_section():
        split_cands = candidates.get("split_candidates", [])
        if len(split_cands) == 0:
            return ""
        lines = [
            "### Split Candidates (Top nodes)",
            "You MUST select node_id from this list (use the EXACT node_id value, NOT the number):"
        ]
        for i, cand in enumerate(split_cands[:10]):
            lines.append(
                f"[{i+1}] node_id='{cand['node_id']}' (incident_edges={cand['incident_edges']}, "
                f"out={cand['out_edges']}, in={cand['in_edges']}, "
                f"currencies={cand['currency_top3']})"
            )
        return "\n".join(lines) + "\n"

    def build_adjust_section():
        adjust_cands = candidates.get("adjust_candidates", [])
        if len(adjust_cands) == 0:
            return ""
        lines = [
            "### Adjust Candidates",
            "**Modify transaction timing and amounts to change temporal and financial patterns.**",
            "You MUST select edge_id from this list (use the EXACT edge_id value, NOT the number):"
        ]
        for i, cand in enumerate(adjust_cands[:10]):
            lines.append(
                f"[{i+1}] edge_id='{cand['edge_id']}' (from={cand['from']}, to={cand['to']}, "
                f"amount={cand['amount_paid']}, risk_score={cand['score_s6_risk']:.3f})"
            )
        return "\n".join(lines) + "\n"

    # Build candidates sections in the order of allowed_tools
    tool_builders = {
        "inject_intermediary": build_inject_section,
        "merge_accounts": build_merge_section,
        "split_account": build_split_section,
        "adjust_transaction": build_adjust_section,
    }

    for tool in allowed_tools:
        if tool in tool_builders:
            section = tool_builders[tool]()
            if section:
                candidates_section += section

    # Build allowed parameters section
    params_section = ""
    if "inject_intermediary" in allowed_tools:
        inject_params = allowed_params.get("inject", {})
        params_section += f"""### inject_intermediary
- depth: {inject_params.get('depth', [1, 2])}
- time_delta_seconds: {inject_params.get('time_delta_seconds', [1, 5, 60])}
- max_edge_ids: {inject_params.get('max_edge_ids', 3)}

"""

    if "merge_accounts" in allowed_tools:
        merge_params = allowed_params.get("merge", {})
        params_section += f"""### merge_accounts
- drop_self_loops: {merge_params.get('drop_self_loops', [False, True])}
- max_pairs: {merge_params.get('max_pairs', 2)}

"""

    if "split_account" in allowed_tools:
        split_params = allowed_params.get("split", {})
        params_section += f"""### split_account
- split_ratio: {split_params.get('split_ratio', [0.2, 0.3, 0.4])}
- move_direction: {split_params.get('move_direction', ['out', 'in', 'both'])}
- edge_sampling: {split_params.get('edge_sampling', ['time_stratified', 'random_within_currency', 'random'])}
- max_node_ids: {split_params.get('max_node_ids', 2)}

"""

    if "adjust_transaction" in allowed_tools:
        adjust_params = allowed_params.get("adjust", {})
        params_section += f"""### adjust_transaction
- time_offset_seconds: {adjust_params.get('time_offset_seconds', [345600, 432000, -345600, -432000])} (4-5 days forward/backward)
- amount_multiplier: {adjust_params.get('amount_multiplier', [0.95, 0.98, 1.02, 1.05])}
- max_edge_ids: {adjust_params.get('max_edge_ids', 3)}

"""

    # Build last failure section
    failure_section = ""
    if last_failure:
        failure_section = f"""## Last Failure
- Reason: {last_failure}

"""

    # Build allowed tools section
    if len(allowed_tools) == 0:
        tools_text = "No tools available. You must return stop."
    else:
        tools_text = f"You may use: {', '.join(allowed_tools)}"

    # Build tool examples section (dynamically based on allowed_tools)
    import random
    tool_examples = {
        "inject_intermediary": '   {"tool":"inject_intermediary","args":{"edge_ids":["row3169173","row3987141"],"depth":1,"time_delta_seconds":5},"rationale":"..."}',
        "merge_accounts": '   {"tool":"merge_accounts","args":{"pairs":[{"a":"24|803B8ACD0","b":"24|803CD14B0"}],"drop_self_loops":false},"rationale":"..."}',
        "split_account": '   {"tool":"split_account","args":{"node_ids":["70|100428738"],"split_ratio":0.3,"move_direction":"both","edge_sampling":"random_within_currency"},"rationale":"..."}',
        "adjust_transaction": '   {"tool":"adjust_transaction","args":{"edge_ids":["row3169173"],"time_offset_seconds":345600,"amount_multiplier":0.98},"rationale":"..."}',
    }

    # Shuffle allowed_tools to randomize order in prompt
    shuffled_tools = allowed_tools.copy()
    random.shuffle(shuffled_tools)

    # Build examples section
    examples_lines = []
    for idx, tool in enumerate(shuffled_tools, 1):
        if tool in tool_examples:
            examples_lines.append(f"{idx}. {tool}:")
            examples_lines.append(tool_examples[tool])
            examples_lines.append("")

    # Always add stop at the end
    examples_lines.append(f"{len(shuffled_tools) + 1}. stop:")
    examples_lines.append('   {"tool":"stop","args":{},"rationale":"..."}')

    examples_section = "\n".join(examples_lines)

    # Build the complete prompt
    prompt = f"""# AML Transaction Graph Perturbation Task

## Objective
Your goal is to transform this money laundering network to make it MORE COMPLEX and HARDER TO DETECT.
You should:
- Increase the number of intermediary nodes and transactions (more complex network structure)
- Reduce detection scores by evading anti-money laundering (AML) rules
- Create more sophisticated laundering patterns that are harder to trace

## Detection Mechanism
The AML system evaluates your network using multiple rules:
New accounts with immediate large outflows
Rapid inflow followed by quick outflow
Fast in-fast out (balanced inflow/outflow with low balance)
Aim to reduce rule hits while maintaining network functionality.

## Available Tools and Their Effects

### inject_intermediary
**Purpose**: Add intermediary nodes between existing transactions to create longer laundering chains
**Effect**: Increases network complexity, dilutes direct connections, may reduce S3/S6 detection
**When to use**: Break up direct suspicious transfers, add layers to the network

### merge_accounts
**Purpose**: Combine multiple accounts into one, consolidating their transaction history
**Effect**: Reduces node count, may create self-loops (removed), can reduce S1/S7 detection
**When to use**: Merge similar accounts to simplify patterns, reduce redundant nodes

### split_account
**Purpose**: Split one account into multiple accounts, distributing its transactions
**Effect**: Increases node count, diversifies transaction patterns, may reduce S5/S8 detection
**When to use**: Break up high-degree nodes, distribute concentrated activity

### adjust_transaction
**Purpose**: Modify transaction timing and amounts to alter temporal/financial patterns
**Effect**: Changes time-based and amount-based features, may reduce S6/S7 detection
**When to use**: Break periodic patterns, adjust timing to avoid rapid in-out detection

---

You are perturbing cluster {cluster_id}, step {step_idx}.

## Cluster Summary
- Nodes: {cluster_summary['n_nodes']}
- Internal laundering edges: {cluster_summary['laundering_edge_count_internal']}
- Max in-degree: {cluster_summary['max_in_deg']}
- Max out-degree: {cluster_summary['max_out_deg']}
- Avg degree: {cluster_summary['avg_deg']:.2f}
- Time span: {cluster_summary['time_span_seconds']} seconds
- Top currencies: {', '.join(cluster_summary['currency_top3'])}

## Budget Remaining
- Steps left: {budget_left['steps_left']}
- New nodes left: {budget_left['new_nodes_left']}
- New edges left: {budget_left['new_edges_left']}
- Merges left: {budget_left['merges_left']}
- Splits left: {budget_left['splits_left']}

## Operation History
- Inject count: {history.get('inject_count', 0)}
- Merge count: {history.get('merge_count', 0)}
- Split count: {history.get('split_count', 0)}

{failure_section}## Allowed Tools
{tools_text}

## Candidates

{candidates_section}## Allowed Parameters

{params_section}## Instructions

**Strategy Tips:**
1. **Prioritize complexity**: Use inject_intermediary to add layers and make the network harder to trace
2. **Balance operations**: Mix different tools to create diverse patterns
3. **Target high-risk transactions**: Use adjust_transaction on transactions with high risk scores
4. **Think long-term**: Early steps should focus on building complexity, later steps on fine-tuning

You must output ONLY valid JSON with one of these formats:

{examples_section}

CRITICAL RULES:
- You MUST use the EXACT edge_id / node_id / pair values (e.g., 'row3169173', '70|100428738') from the candidates lists above
- Do NOT use the list numbers (e.g., [1], [2]) as IDs - those are just for reference
- Copy the EXACT ID strings including all characters (numbers, letters, pipes |)
- Do NOT make up or guess IDs that are not in the candidates list
- Output ONLY valid JSON, no additional text before or after
- Use exact parameter values from allowed parameters
"""

    return prompt
