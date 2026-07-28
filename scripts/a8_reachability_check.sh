#!/usr/bin/env bash
# A8 reachability: how many whitelist / holdout-positive drugs have NO path to any
# DIS node in the FILTERED graph (DRG-DIS removed -> only drug->gene/pathway/phenotype->...->DIS).
#
# Primary answer uses pure Cypher (shortestPath, bounded at K hops) so it runs on stock
# neo4j:5-community with no plugins. The exact any-distance answer (GDS WCC) is at the bottom,
# guarded by an availability check.
#
# Traversal is UNDIRECTED (-[*]-): the 13 undirected labels are stored one-way (G-A3-8), so a
# directed pattern silently misses half of them. shortestPath does bidirectional BFS and
# short-circuits on the first DIS, which is what keeps the bounded query cheap.
#
# Run from repo root:  bash scripts/a8_reachability_check.sh
set -euo pipefail

CONT=retracerx-neo4j
NEO_USER=neo4j
NEO_PASS=retracerx
K=4          # max hops. drug->gene->disease = 2; ->gene->gene->disease = 3; pathway-mediated = 4.
             # Bounded K OVER-counts no-path (a drug reachable only at >K hops reads as no-path),
             # so bounded_no_path >= true_no_path >= edgeless. Raise K to tighten; lower if slow.

# --- stage the two id lists where LOAD CSV can read them (import dir) ---
# A0 quirk: the entrypoint reclaims neo4j/import as uid 7474 on docker start; chown back to write.
sudo chown -R "$(id -u):$(id -g)" neo4j/import
cp data/holdout/candidate_whitelist.csv neo4j/import/reach_wl.csv
cp data/holdout/holdout_positives.csv   neo4j/import/reach_ho.csv
chmod 644 neo4j/import/reach_wl.csv neo4j/import/reach_ho.csv

run() { docker exec -i "$CONT" cypher-shell -u "$NEO_USER" -p "$NEO_PASS" --format plain "$1"; }

echo "== bounded reachability (<= ${K} hops), UNDIRECTED, target label :DIS =="

# --- whitelist ---
run "
LOAD CSV WITH HEADERS FROM 'file:///reach_wl.csv' AS row
WITH DISTINCT row.okg_drug_id AS did
MATCH (d:Node {id: did})
OPTIONAL MATCH p = shortestPath( (d)-[*1..${K}]-(:DIS) )
RETURN 'whitelist'                                   AS population,
       count(d)                                      AS drugs,
       sum(CASE WHEN p IS NULL THEN 1 ELSE 0 END)    AS no_path_to_DIS,
       round(100.0 * sum(CASE WHEN p IS NULL THEN 1 ELSE 0 END) / count(d), 1) AS pct;
"

# --- holdout positives (frozen disease-cold bundle; 466 distinct drugs) ---
run "
LOAD CSV WITH HEADERS FROM 'file:///reach_ho.csv' AS row
WITH DISTINCT row.okg_drug_id AS did
MATCH (d:Node {id: did})
OPTIONAL MATCH p = shortestPath( (d)-[*1..${K}]-(:DIS) )
RETURN 'holdout_positives'                           AS population,
       count(d)                                      AS drugs,
       sum(CASE WHEN p IS NULL THEN 1 ELSE 0 END)    AS no_path_to_DIS,
       round(100.0 * sum(CASE WHEN p IS NULL THEN 1 ELSE 0 END) / count(d), 1) AS pct;
"

echo "== sanity ordering: edgeless (86 / 13) <= no-path <= drugs. no-path >= edgeless must hold. =="

# --- optional: write the no-path drug ids out for the writeup (whitelist example) ---
echo "== writing whitelist no-path drug ids -> neo4j/import/reach_wl_nopath.csv =="
run "
LOAD CSV WITH HEADERS FROM 'file:///reach_wl.csv' AS row
WITH DISTINCT row.okg_drug_id AS did
MATCH (d:Node {id: did})
OPTIONAL MATCH p = shortestPath( (d)-[*1..${K}]-(:DIS) )
WITH d WHERE p IS NULL
RETURN d.id AS okg_drug_id, coalesce(d.name, d.id) AS name
ORDER BY okg_drug_id;
" > neo4j/import/reach_wl_nopath.csv || true
echo "   (also run the same block against splits/{drug_cold,random}_holdout_positives.csv)"

cat <<'NOTE'

== EXACT any-distance answer (requires the GDS plugin; skip if not installed) ==
Bounded K answers "no SHORT mechanistic path". The literal "no path at ANY distance" is a
connected-components question: a drug can reach some DIS iff its component contains a DIS node.
If GDS is available:

  CALL gds.graph.project('f', '*', { r: { type: '*', orientation: 'UNDIRECTED' } });

  CALL gds.wcc.stream('f') YIELD nodeId, componentId
  WITH componentId, gds.util.asNode(nodeId) AS n
  WITH componentId,
       sum(CASE WHEN n:DIS THEN 1 ELSE 0 END) AS dis_in_component,
       collect(CASE WHEN n:DRG THEN n.id END)  AS drug_ids
  WHERE dis_in_component = 0                    // components with no disease at all
  UNWIND drug_ids AS okg_drug_id
  WITH collect(okg_drug_id) AS no_path_drugs    // every drug that cannot reach ANY DIS
  LOAD CSV WITH HEADERS FROM 'file:///reach_wl.csv' AS row
  WITH no_path_drugs, collect(DISTINCT row.okg_drug_id) AS wl
  RETURN size([x IN wl WHERE x IN no_path_drugs]) AS whitelist_no_path_exact,
         size(wl)                                  AS whitelist_drugs;

  CALL gds.graph.drop('f');

This is exact and far cheaper than path enumeration for the any-distance question. It should
nearly coincide with the bounded count; any gap is drugs reachable only at > K hops.
NOTE
