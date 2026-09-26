"""The 7 complete local SCAND recordings and their short IDs."""
from pathlib import Path

SCAND_ROOT = Path(__file__).resolve().parents[3]
RAW = SCAND_ROOT / "raw"

RECORDINGS = {
    "Brackenridge": "A_Jackal_Brackenridge_Greg_Sat_Nov_13_94",
    "Butler": "A_Spot_Butler_LBJ_Sat_Nov_13_104",
    "JCL": "A_Spot_JCL_JCL_Wed_Nov_10_65",
    "Library_MLK": "A_Spot_Library_MLK_Thu_Nov_18_122",
    "Sanjac": "B_Jackal_Sanjac_Stadium_Sat_Nov_13_89",
    "GDC": "B_Spot_GDC_AHG_Mon_Nov_15_116",
    "RLM": "B_Spot_RLM_RLM_Wed_Nov_10_45",
}


def robot(rec: str) -> str:
    return "jackal" if "_Jackal_" in RECORDINGS[rec] else "spot"


def bag_path(rec: str) -> Path:
    return RAW / f"{RECORDINGS[rec]}.bag"
