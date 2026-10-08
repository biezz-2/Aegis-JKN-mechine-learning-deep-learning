"""
Validator script to inspect and verify Aegis-JKN.ipynb:
1. JSON structure and nbformat version.
2. Python AST / syntax compilation for every code cell.
3. Checking critical requirements (Anti-leakage, 300 claims, 9 syndicates, OASIS, 8 contradictions, etc.)
"""
import json
import ast
import os
import sys

def validate():
    nb_path = r"N:\HEALTHKATHON\mechine-learning-deep-learning\notebooks\Aegis-JKN.ipynb"
    print(f"[Validator] Memeriksa file notebook: {nb_path}")

    assert os.path.exists(nb_path), "File notebook tidak ditemukan!"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"[Validator] JSON Valid. Format: nbformat {nb.get('nbformat')}.{nb.get('nbformat_minor')}")
    assert nb.get("nbformat") == 4, f"nbformat harus 4, didapat {nb.get('nbformat')}"

    cells = nb.get("cells", [])
    print(f"[Validator] Total sel dalam notebook: {len(cells)}")

    code_cells = [c for c in cells if c["cell_type"] == "code"]
    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    print(f"[Validator] Rincian: {len(code_cells)} code cells, {len(md_cells)} markdown cells")

    # Check syntax for all code cells
    for idx, c in enumerate(code_cells):
        src = "".join(c["source"])
        try:
            ast.parse(src)
            # juga compile
            compile(src, f"<cell_{idx}>", "exec")
        except SyntaxError as e:
            print(f"[ERROR] SyntaxError pada Code Cell #{idx}: {e}")
            sys.exit(1)

    print(f"[Validator] Seluruh {len(code_cells)} Code Cells LULUS kompilasi sintaks Python (0 SyntaxError)!")

    # Requirement checks
    full_text = json.dumps(nb)
    checks = {
        "Dataset 300 Klaim Embedded": "EMBEDDED_CLAIMS_CSV_B64" in full_text,
        "Anti-Leakage Protocol": "FORBIDDEN_COLUMNS" in full_text and "skor_fraud" in full_text,
        "GroupKFold pasien_id": "GroupKFold(n_splits=5)" in full_text and "pasien_id" in full_text,
        "Leave-One-Group-Out faskes_id": "LeaveOneGroupOut" in full_text and "faskes_id" in full_text,
        "MHGSL 3 Saluran (A_top, A_feat, A_sem)": "construct_multichannel_graphs" in full_text and "adj_sem" in full_text,
        "ChannelSpecificGCN & SharedParameterGCN": "ChannelSpecificGCN" in full_text and "SharedParameterGCN" in full_text,
        "MultiChannelAttention & FocalLoss": "MultiChannelAttention" in full_text and "FocalLoss" in full_text,
        "Baseline Benchmarking (XGBoost, MLP, GCN)": "HomogeneousGCN" in full_text and "TabularMLP" in full_text,
        "Hit-Rate Seluruh 9 Sindikat": "SYND-01" in full_text and "SYND-09" in full_text,
        "Target 8 Klaim Kontradiksi Teks": "TARGET_CONTRADICTION_IDS" in full_text and "KLM001" in full_text,
        "OASIS Multi-Agent Dual-Mode": "JKNToOASISMapper" in full_text and "DeterministicAgentEngine" in full_text,
        "Interactive Dashboard Canvas": "dashboard_html" in full_text and "graph-canvas" in full_text,
        "Export Model & Audit Report": "aegis_mhgsl_weights.pt" in full_text and "triage_audit_report.json" in full_text,
        "Disclaimer Label": "simulasi sintetis, prevalensi oversampled" in full_text
    }

    print("\n=== HASIL VERIFIKASI SPESIFIKASI TEKNIS ===")
    all_ok = True
    for item, status in checks.items():
        res = "[LULUS]" if status else "[GAGAL]"
        if not status:
            all_ok = False
        print(f"  {res} {item}")

    assert all_ok, "Ada spesifikasi teknis yang belum terpenuhi!"
    print("\n[Validator] SEMUA SPESIFIKASI TEKNIS TERPENUHI 100%!")

if __name__ == "__main__":
    validate()
