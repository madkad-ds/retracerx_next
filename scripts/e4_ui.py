#!/usr/bin/env python3
"""
e4_ui.py — ReTraceRx single-disease explorer (E4).

Landing view: type a disease → ranked repurposing candidates; per candidate the
cited-path GRAPH and the TRIPLES + SOURCES sit side by side. A toggle shows the
ungrounded baseline for contrast (confident picks, empty citations).

Talks to the E2 API server-side (UI → API → agent); never imports the agent.
Launch (serve_env sourced; API already running on :8080):
    streamlit run e4_ui.py --server.port 8501

Binds to the REAL contract shapes:
  contract  : {query_disease, candidates:[{drug_graph_id, drug_name, score,
              support_count, specificity, paths:[{path_id, triples[[s,p,o]],
              nl_evidence, sources[]}]}]}
  grounded  : {abstain, ranking:[{drug_graph_id, drug_name, confidence, supported,
              faithful, cited_path_ids[], hallucinated_path_ids[]}], _meta}
  ungrounded: {picks:[{drug_graph_id, cited_path_ids:[]}], proposed_names[],
              ranked_drug_ids[]}
"""
from __future__ import annotations
import os
import httpx
import streamlit as st
from pyvis.network import Network

API = os.environ.get("RETRACERX_API_BASE", "http://localhost:8080")
SPLITS = ["disease_cold", "random", "drug_cold"]   # headline first
NODE_COLOR = {"disease": "#c1432b", "gene": "#2f8f5b", "drug": "#2b5bc1", "entity": "#7a7a7a"}

st.set_page_config(page_title="ReTraceRx", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .rx-badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:0.72rem;
            font-weight:600;letter-spacing:.02em;margin-right:6px;vertical-align:middle}
  .rx-ok{background:#12351f;color:#7ee2a8;border:1px solid #1f5c37}
  .rx-warn{background:#3a1414;color:#ff9e9e;border:1px solid #7a2323}
  .rx-mut{background:#23262d;color:#9aa2ad;border:1px solid #333842}
  .rx-src{display:inline-block;background:#20242b;color:#b7c0cc;border:1px solid #333842;
          border-radius:6px;padding:1px 7px;font-size:.72rem;margin:2px 4px 2px 0}
  .rx-conf{font-variant-numeric:tabular-nums;font-weight:700}
  .rx-triple{font-family:ui-monospace,Menlo,monospace;font-size:.8rem;line-height:1.5}
  .rx-node-dis{color:#e0674f}.rx-node-gen{color:#49b97b}.rx-node-drg{color:#5b83e8}
</style>""", unsafe_allow_html=True)


# ── API helpers (deterministic responses → safe to cache) ────────────────────
def _post(path, payload, timeout=300):
    r = httpx.post(f"{API}{path}", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def link_disease(text): return _post("/link", {"disease_text": text}, timeout=60)

@st.cache_data(show_spinner=False)
def get_contract(gid, split):
    return _post("/contract", {"disease_graph_id": gid, "split": split})

@st.cache_data(show_spinner=False)
def get_ranking(gid, split, grounded, variant="free"):
    return _post("/repurpose", {"disease_graph_id": gid, "split": split,
                                "grounded": grounded, "ungrounded_variant": variant})


# ── graph helpers (honest: ids + relation types + serializer's nl sentence) ──
def node_type(nid: str) -> str:
    if nid.startswith("MONDO"):   return "disease"
    if nid.startswith("ENSG"):    return "gene"
    if nid.startswith(("CHEMBL", "DRUGBANK")): return "drug"
    return "entity"

def humanize(pred: str) -> str:
    tail = pred.split("__")[-1] if "__" in pred else pred
    return tail.replace("_", " ").lower()

def build_graph_html(paths, disease_label, drug_label, drug_gid, height=430):
    net = Network(height=f"{height}px", width="100%", directed=True,
                  bgcolor="#0e1117", font_color="#d6dbe2", cdn_resources="remote")
    net.toggle_physics(True)
    seen = set()
    def add(nid):
        if nid in seen: return
        t = node_type(nid)
        label = drug_label if (nid == drug_gid and drug_label) else \
                (disease_label if t == "disease" and disease_label else nid)
        net.add_node(nid, label=label, color=NODE_COLOR[t], shape="dot",
                     title=f"{t}: {nid}")
        seen.add(nid)
    for p in paths:
        for s, pred, o in p["triples"]:
            add(s); add(o)
            net.add_edge(s, o, label=humanize(pred), title=pred, color="#4a5160")
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        return net.generate_html()

def node_span(nid):
    cls = {"disease": "rx-node-dis", "gene": "rx-node-gen", "drug": "rx-node-drg"}.get(node_type(nid), "")
    return f"<span class='{cls}'>{nid}</span>"

def render_html(html, height=460):
    """Embed the pyvis document in a sandboxed iframe. Prefer st.iframe (the
    non-deprecated path in Streamlit ≥1.61); fall back to components.html."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)              # accepts raw HTML content
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)

@st.cache_data(show_spinner=False)
def graph_html_cached(key, _paths, disease_label, drug_label, drug_gid):
    """Cache the built HTML across reruns. `key` (the cited path_ids tuple) is the
    hash key; `_paths` is ignored for hashing (path_ids fully determine it)."""
    return build_graph_html(_paths, disease_label, drug_label, drug_gid)


# ── header ───────────────────────────────────────────────────────────────────
st.title("ReTraceRx — drug-repurposing explorer")
st.caption("Every grounded pick cites verifiable paths in the knowledge graph, or the agent abstains. "
           "The ungrounded baseline answers confidently with nothing to cite.")

c1, c2, c3 = st.columns([4, 2, 1])
disease_text = c1.text_input("Disease", value="type 2 diabetes",
                             label_visibility="collapsed", placeholder="e.g. type 2 diabetes")
split = c2.selectbox("Split (KGE scoring)", SPLITS, index=0, label_visibility="collapsed")
go = c3.button("Explore", type="primary", use_container_width=True)

# health hint
try:
    h = httpx.get(f"{API}/health", timeout=10).json()
    if not h.get("ok"):
        st.warning("API not ready yet.")
    for w in h.get("warnings", []):
        st.info(w)
except Exception:
    st.error(f"Cannot reach the API at {API}. Start it: `uvicorn e2_api:app --port 8080`")
    st.stop()

if "gid" not in st.session_state:
    st.session_state.gid = None

if go and disease_text.strip():
    q = link_disease(disease_text.strip())
    if q.get("graph_id"):
        st.session_state.gid = q["graph_id"]
        st.session_state.linkscore = q.get("link_score")
        st.session_state.dlabel = disease_text.strip()
    else:
        # low-confidence link (D-B7): disambiguate rather than guess
        st.session_state.gid = None
        st.session_state.alts = q.get("alternatives", [])
        st.session_state.dlabel = disease_text.strip()

# disambiguation panel
if st.session_state.gid is None and st.session_state.get("alts"):
    st.subheader("Which disease did you mean?")
    for a in st.session_state.alts[:8]:
        aid = a.get("graph_id") or a.get("curie")
        if st.button(f"{a.get('name', aid)}  ·  {aid}", key=f"alt_{aid}"):
            st.session_state.gid = aid
            st.rerun()

gid = st.session_state.gid
if not gid:
    st.stop()

# ── fetch grounded ranking + contract, build path index ─────────────────────
dlabel = st.session_state.get("dlabel", gid)
with st.spinner("Retrieving, reasoning, and verifying over the graph…"):
    grounded = get_ranking(gid, split, True)
    contract = get_contract(gid, split)

paths_by_id, cand_meta = {}, {}
for c in contract.get("candidates", []):
    cand_meta[c["drug_graph_id"]] = c
    for p in c.get("paths", []):
        paths_by_id[p["path_id"]] = p

st.markdown(f"### {dlabel}  ·  `{gid}`  ·  link score "
            f"{st.session_state.get('linkscore', contract.get('query_disease', {}).get('link_score', '—'))}")

if grounded.get("abstain"):
    st.warning(f"The agent abstained for this disease. "
               f"Reason: {grounded.get('abstain_reason') or 'no verifiable candidate'}")
    st.stop()

ranking = grounded.get("ranking", [])
st.markdown(f"**{len(ranking)} grounded candidates** · split `{split}` · "
            f"{grounded.get('_meta', {}).get('latency_s', '—')}s")

def _badge(cand):
    if cand.get("hallucinated_path_ids"):
        return "<span class='rx-badge rx-warn'>⚠ fabricated citation caught</span>"
    if cand.get("faithful") and cand.get("supported"):
        return "<span class='rx-badge rx-ok'>✓ verified citations</span>"
    return "<span class='rx-badge rx-mut'>abstained at pick</span>"

for i, cand in enumerate(ranking, 1):
    dgid = cand.get("drug_graph_id"); name = cand.get("drug_name") or dgid
    meta = cand_meta.get(dgid, {})
    cited = cand.get("cited_path_ids", []) or []

    head = (f"**{i}. {name}**  ·  <span class='rx-conf'>conf {cand.get('confidence')}</span>  "
            f"{_badge(cand)}"
            f"<span class='rx-badge rx-mut'>{len(cited)} paths</span>")
    if meta:
        head += (f"<span class='rx-badge rx-mut'>support {meta.get('support_count','—')}</span>"
                 f"<span class='rx-badge rx-mut'>score {round(meta.get('score', 0), 3)}</span>")
    st.markdown(head, unsafe_allow_html=True)

    # per-candidate evidence dropdown. Graph HTML is cached on the cited path_ids
    # (built once) and vis.js loads from CDN (~11 KB), so opening these is cheap.
    with st.expander("Evidence", expanded=(i == 1)):
        cited_paths = [paths_by_id[pid] for pid in cited if pid in paths_by_id]
        gcol, tcol = st.columns(2)
        with gcol:
            if cited_paths:
                html = graph_html_cached(tuple(cited), cited_paths, dlabel, name, dgid)
                render_html(html, height=460)
            else:
                st.info("No resolvable cited paths.")
        with tcol:
            for p in cited_paths:
                st.markdown(f"*{p.get('nl_evidence','')}*")
                lines = [f"{node_span(s)} —<b>{humanize(pred)}</b>→ {node_span(o)}"
                         for s, pred, o in p["triples"]]
                st.markdown("<div class='rx-triple'>" + "<br>".join(lines) + "</div>",
                            unsafe_allow_html=True)
                srcs = "".join(f"<span class='rx-src'>{s}</span>" for s in p.get("sources", []))
                st.markdown(f"<div>{srcs}</div>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:8px 0;border-color:#23262d'>", unsafe_allow_html=True)
            missing = [pid for pid in cited if pid not in paths_by_id]
            if missing:
                st.caption(f"Unresolved path ids: {', '.join(missing)}")

# ── ungrounded contrast ──────────────────────────────────────────────────────
st.divider()
if st.toggle("Show ungrounded baseline (same disease, no graph)"):
    with st.spinner("Asking the model with no graph access…"):
        ung = get_ranking(gid, split, False)
    picks = ung.get("picks", [])
    proposed = ung.get("proposed_names", [])
    st.markdown("**Ungrounded baseline** — confident picks, "
                "<span class='rx-badge rx-warn'>uncited by construction</span>",
                unsafe_allow_html=True)
    st.caption("This system never saw the graph, so it has no paths to cite. "
               "That is the failure grounding removes — not a bug in the baseline.")
    if proposed:
        st.write("Model proposed:", ", ".join(proposed))
    for p in picks[:15]:
        dgid = p.get("drug_graph_id")
        nm = p.get("drug_name") or p.get("proposed_name") or dgid
        st.markdown(f"- **{nm}** (`{dgid}`) · "
                    f"<span class='rx-src'>0 citations</span>", unsafe_allow_html=True)
