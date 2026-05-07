"""HSAE GeoAgent Tools — Demo Script
=====================================
Demonstrates all 9 HSAE tools for transboundary water law compliance.

Run standalone (no QGIS, no hydrosovereign required):

    python examples/hsae_demo.py

Or from the GeoAgent QGIS plugin Python console:

    exec(open("examples/hsae_demo.py").read())

Install optional dependency for real satellite data:

    pip install hydrosovereign>=6.0.7
"""

from geoagent.tools.hsae import hsae_tools

tools = {t.tool_name: t for t in hsae_tools()}

SEP = "=" * 62

# ── 1. Full compliance analysis ───────────────────────────────────────────────
print(SEP)
print("HSAE GeoAgent Demo — Blue Nile / GERD")
print(SEP)

print("\n[1] analyze_basin_compliance('Blue Nile')")
result = tools["analyze_basin_compliance"]("Blue Nile")
print(f"    Legal status : {result['legal_status']}")
print(f"    Articles     : {result['triggered_articles']}")
print(f"    Recommendation: {result['recommendation']}")
print("    Indices:")
for k, v in result["indices"].items():
    print(f"      {k:12s} = {v}")
print(f"    Source       : {result['source']}")

# ── 2. Individual indices ─────────────────────────────────────────────────────
print("\n[2] Individual index tools")

r_atdi = tools["compute_atdi"]("Blue Nile")
print(f"    compute_atdi   → ATDI = {r_atdi['ATDI_pct']}%  ({r_atdi['legal_status']})")

r_afsf = tools["compute_afsf"]("Blue Nile")
print(f"    compute_afsf   → AFSF = {r_afsf['AFSF_pct']}%  (30-day peak)")

r_ahifd = tools["compute_ahifd"]("Blue Nile")
print(
    f"    compute_ahifd  → AHIFD = {r_ahifd['AHIFD_pct']}%  "
    f"(Art.7 triggered: {r_ahifd['art7_triggered']})"
)

r_atci = tools["compute_atci"]("Blue Nile")
print(
    f"    compute_atci   → ATCI = {r_atci['ATCI_pct']}%  ({r_atci['compliance_level']} compliance)"
)

r_ci = tools["compute_conflict_index"]("Blue Nile")
print(f"    compute_ci     → CI = {r_ci['CI_score']}  ({r_ci['level']})")

r_adts = tools["compute_adts"]("Blue Nile")
print(
    f"    compute_adts   → ADTS = {r_adts['ADTS_pct']}%  "
    f"(Art.9: {r_adts['art9_data_sharing']})"
)

# ── 3. UNWC compliance screen ─────────────────────────────────────────────────
print("\n[3] run_unwc_compliance('Blue Nile')")
r_unwc = tools["run_unwc_compliance"]("Blue Nile")
print(f"    Overall: {r_unwc['overall_compliance']}")
for finding in r_unwc["findings"]:
    print(f"    {finding['article']:8s} {finding['status']:25s} {finding['value']}")

# ── 4. AI negotiation recommendation ─────────────────────────────────────────
print("\n[4] get_negotiation_recommendation('Blue Nile')")
r_neg = tools["get_negotiation_recommendation"]("Blue Nile")
print(f"    P(Negotiation) = {r_neg['p_negotiation_pct']}%")
print(f"    Pathway        : {r_neg['recommended_pathway']}")
print(f"    Description    : {r_neg['pathway_description']}")
print(f"    Disclaimer     : {r_neg['disclaimer']}")

# ── 5. Multi-basin scan ───────────────────────────────────────────────────────
print(f"\n[5] Multi-basin ATDI scan")
print(f"    {'Basin':<20} {'ATDI':>6}  Legal status")
print(f"    {'-'*20} {'-'*6}  {'-'*28}")
for basin in [
    "Blue Nile",
    "Mekong",
    "Euphrates",
    "Tigris",
    "Indus",
    "Danube",
    "Rhine",
    "Zambezi",
]:
    r = tools["compute_atdi"](basin)
    print(f"    {basin:<20} {r['ATDI_pct']:>5.1f}%  {r['legal_status']}")

print(f"\n{SEP}")
print("✅  All 9 HSAE tools functional")
print("    No hydrosovereign or QGIS session required")
print("    Install for real GEE satellite data:")
print("      pip install hydrosovereign>=6.0.7")
print(SEP)
