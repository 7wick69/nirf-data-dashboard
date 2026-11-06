"""
pipeline/make_dummy_data.py

Generates realistic dummy NIRF datasets (2017-2025) for development & testing.
Outputs:
  - data/clean/nirf_long.parquet
  - data/clean/institutions.parquet
  - data/clean/nirf_long.csv (convenience copy)
"""

import os
import random
import numpy as np
import pandas as pd

# Fix random seed for reproducible dummy data
np.random.seed(42)
random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clean")
os.makedirs(DATA_DIR, exist_ok=True)

INSTITUTIONS = [
    {
        "id": "NIRF_001",
        "canonical_name": "Indian Institute of Technology Madras",
        "state": "Tamil Nadu",
        "city": "Chennai",
        "aliases": ["IIT Madras", "IITM", "IIT-M", "Indian Institute of Technology, Madras"],
        "base_rank": 1,
        "rank_trajectory": {2017: 8, 2018: 6, 2019: 5, 2020: 4, 2021: 4, 2022: 3, 2023: 3, 2024: 2, 2025: 2},
        "score_base": 88.5,
    },
    {
        "id": "NIRF_002",
        "canonical_name": "Indian Institute of Science",
        "state": "Karnataka",
        "city": "Bengaluru",
        "aliases": ["IISc", "IISc Bangalore", "IISc Bengaluru", "Indian Institute of Science Bangalore"],
        "base_rank": 1,
        "rank_trajectory": {2017: 1, 2018: 1, 2019: 1, 2020: 1, 2021: 1, 2022: 1, 2023: 1, 2024: 1, 2025: 1},
        "score_base": 91.0,
    },
    {
        "id": "NIRF_003",
        "canonical_name": "Indian Institute of Technology Delhi",
        "state": "Delhi",
        "city": "New Delhi",
        "aliases": ["IIT Delhi", "IITD", "IIT-D", "Indian Institute of Technology, Delhi"],
        "base_rank": 3,
        "rank_trajectory": {2017: 4, 2018: 3, 2019: 3, 2020: 3, 2021: 2, 2022: 2, 2023: 2, 2024: 3, 2025: 3},
        "score_base": 87.2,
    },
    {
        "id": "NIRF_004",
        "canonical_name": "Indian Institute of Technology Bombay",
        "state": "Maharashtra",
        "city": "Mumbai",
        "aliases": ["IIT Bombay", "IITB", "IIT-B", "Indian Institute of Technology, Bombay"],
        "base_rank": 4,
        "rank_trajectory": {2017: 3, 2018: 4, 2019: 4, 2020: 2, 2021: 3, 2022: 4, 2023: 4, 2024: 4, 2025: 4},
        "score_base": 86.8,
    },
    {
        "id": "NIRF_005",
        "canonical_name": "Indian Institute of Technology Kanpur",
        "state": "Uttar Pradesh",
        "city": "Kanpur",
        "aliases": ["IIT Kanpur", "IITK", "Indian Institute of Technology, Kanpur"],
        "base_rank": 5,
        "rank_trajectory": {2017: 5, 2018: 5, 2019: 6, 2020: 6, 2021: 5, 2022: 5, 2023: 5, 2024: 5, 2025: 5},
        "score_base": 82.5,
    },
    {
        "id": "NIRF_006",
        "canonical_name": "Indian Institute of Technology Kharagpur",
        "state": "West Bengal",
        "city": "Kharagpur",
        "aliases": ["IIT Kharagpur", "IIT KGP", "IITKGP"],
        "base_rank": 6,
        "rank_trajectory": {2017: 2, 2018: 2, 2019: 2, 2020: 5, 2021: 6, 2022: 6, 2023: 6, 2024: 6, 2025: 6},
        "score_base": 81.3,
    },
    {
        "id": "NIRF_007",
        "canonical_name": "Indian Institute of Technology Roorkee",
        "state": "Uttarakhand",
        "city": "Roorkee",
        "aliases": ["IIT Roorkee", "IITR"],
        "base_rank": 7,
        "rank_trajectory": {2017: 9, 2018: 8, 2019: 8, 2020: 7, 2021: 7, 2022: 7, 2023: 7, 2024: 7, 2025: 7},
        "score_base": 78.4,
    },
    {
        "id": "NIRF_008",
        "canonical_name": "Indian Institute of Technology Guwahati",
        "state": "Assam",
        "city": "Guwahati",
        "aliases": ["IIT Guwahati", "IITG"],
        "base_rank": 8,
        "rank_trajectory": {2017: 10, 2018: 9, 2019: 9, 2020: 8, 2021: 8, 2022: 8, 2023: 8, 2024: 8, 2025: 8},
        "score_base": 75.9,
    },
    {
        "id": "NIRF_009",
        "canonical_name": "All India Institute of Medical Sciences New Delhi",
        "state": "Delhi",
        "city": "New Delhi",
        "aliases": ["AIIMS", "AIIMS Delhi", "AIIMS New Delhi"],
        "base_rank": 9,
        "rank_trajectory": {2017: 12, 2018: 11, 2019: 10, 2020: 9, 2021: 9, 2022: 9, 2023: 9, 2024: 9, 2025: 9},
        "score_base": 73.8,
    },
    {
        "id": "NIRF_010",
        "canonical_name": "Jawaharlal Nehru University",
        "state": "Delhi",
        "city": "New Delhi",
        "aliases": ["JNU", "Jawaharlal Nehru Univ"],
        "base_rank": 10,
        "rank_trajectory": {2017: 6, 2018: 7, 2019: 7, 2020: 10, 2021: 10, 2022: 10, 2023: 10, 2024: 10, 2025: 10},
        "score_base": 71.5,
    },
    {
        "id": "NIRF_011",
        "canonical_name": "Banaras Hindu University",
        "state": "Uttar Pradesh",
        "city": "Varanasi",
        "aliases": ["BHU", "Banaras Hindu Univ"],
        "base_rank": 11,
        "rank_trajectory": {2017: 7, 2018: 10, 2019: 11, 2020: 11, 2021: 11, 2022: 11, 2023: 11, 2024: 11, 2025: 11},
        "score_base": 69.2,
    },
    {
        "id": "NIRF_012",
        "canonical_name": "Jamia Millia Islamia",
        "state": "Delhi",
        "city": "New Delhi",
        "aliases": ["JMI", "Jamia"],
        "base_rank": 12,
        "rank_trajectory": {2017: 19, 2018: 17, 2019: 15, 2020: 14, 2021: 13, 2022: 12, 2023: 12, 2024: 12, 2025: 12},
        "score_base": 67.8,
    },
    {
        "id": "NIRF_013",
        "canonical_name": "Manipal Academy of Higher Education",
        "state": "Karnataka",
        "city": "Manipal",
        "aliases": ["MAHE", "Manipal University", "Manipal"],
        "base_rank": 13,
        "rank_trajectory": {2017: 18, 2018: 16, 2019: 14, 2020: 13, 2021: 14, 2022: 13, 2023: 13, 2024: 13, 2025: 13},
        "score_base": 66.4,
    },
    {
        "id": "NIRF_014",
        "canonical_name": "Vellore Institute of Technology",
        "state": "Tamil Nadu",
        "city": "Vellore",
        "aliases": ["VIT", "VIT Vellore"],
        "base_rank": 14,
        "rank_trajectory": {2017: 22, 2018: 20, 2019: 18, 2020: 16, 2021: 15, 2022: 14, 2023: 14, 2024: 14, 2025: 14},
        "score_base": 65.1,
    },
    {
        "id": "NIRF_015",
        "canonical_name": "University of Hyderabad",
        "state": "Telangana",
        "city": "Hyderabad",
        "aliases": ["UoH", "Univ of Hyderabad", "Hyderabad Central University", "HCU"],
        "base_rank": 15,
        "rank_trajectory": {2017: 14, 2018: 13, 2019: 13, 2020: 15, 2021: 16, 2022: 15, 2023: 15, 2024: 15, 2025: 15},
        "score_base": 64.0,
    },
    {
        "id": "NIRF_016",
        "canonical_name": "Anna University",
        "state": "Tamil Nadu",
        "city": "Chennai",
        "aliases": ["Anna Univ"],
        "base_rank": 16,
        "rank_trajectory": {2017: 13, 2018: 14, 2019: 16, 2020: 17, 2021: 17, 2022: 16, 2023: 16, 2024: 16, 2025: 16},
        "score_base": 63.2,
    },
    {
        "id": "NIRF_017",
        "canonical_name": "Jadavpur University",
        "state": "West Bengal",
        "city": "Kolkata",
        "aliases": ["JU", "Jadavpur"],
        "base_rank": 17,
        "rank_trajectory": {2017: 15, 2018: 15, 2019: 17, 2020: 18, 2021: 18, 2022: 17, 2023: 17, 2024: 17, 2025: 17},
        "score_base": 62.5,
    },
    {
        "id": "NIRF_018",
        "canonical_name": "National Institute of Technology Tiruchirappalli",
        "state": "Tamil Nadu",
        "city": "Tiruchirappalli",
        "aliases": ["NIT Trichy", "NITT", "NIT Tiruchirappalli"],
        "base_rank": 18,
        "rank_trajectory": {2017: 21, 2018: 19, 2019: 19, 2020: 19, 2021: 19, 2022: 18, 2023: 18, 2024: 18, 2025: 18},
        "score_base": 61.9,
    },
    {
        "id": "NIRF_019",
        "canonical_name": "National Institute of Technology Karnataka",
        "state": "Karnataka",
        "city": "Surathkal",
        "aliases": ["NIT Surathkal", "NITK", "NITK Surathkal"],
        "base_rank": 19,
        "rank_trajectory": {2017: 25, 2018: 24, 2019: 22, 2020: 21, 2021: 20, 2022: 19, 2023: 19, 2024: 19, 2025: 19},
        "score_base": 61.0,
    },
    {
        "id": "NIRF_020",
        "canonical_name": "Amrita Vishwa Vidyapeetham",
        "state": "Tamil Nadu",
        "city": "Coimbatore",
        "aliases": ["Amrita", "Amrita University"],
        "base_rank": 20,
        "rank_trajectory": {2017: 23, 2018: 21, 2019: 20, 2020: 20, 2021: 21, 2022: 20, 2023: 20, 2024: 20, 2025: 20},
        "score_base": 60.5,
    },
    {
        "id": "NIRF_021",
        "canonical_name": "Birla Institute of Technology and Science - Pilani",
        "state": "Rajasthan",
        "city": "Pilani",
        "aliases": ["BITS Pilani", "BITS", "BITS-Pilani"],
        "base_rank": 21,
        "rank_trajectory": {2017: 24, 2018: 23, 2019: 23, 2020: 22, 2021: 22, 2022: 21, 2023: 21, 2024: 21, 2025: 21},
        "score_base": 59.8,
    },
    {
        "id": "NIRF_022",
        "canonical_name": "Indian Institute of Information Technology Hyderabad",
        "state": "Telangana",
        "city": "Hyderabad",
        "aliases": ["IIIT Hyderabad", "IIITH", "IIIT-H"],
        "base_rank": 22,
        "rank_trajectory": {2017: 35, 2018: 32, 2019: 28, 2020: 26, 2021: 24, 2022: 23, 2023: 22, 2024: 22, 2025: 22},
        "score_base": 58.7,
    },
    {
        "id": "NIRF_023",
        "canonical_name": "Thapar Institute of Engineering and Technology",
        "state": "Punjab",
        "city": "Patiala",
        "aliases": ["TIET", "Thapar", "Thapar University"],
        "base_rank": 23,
        "rank_trajectory": {2017: 28, 2018: 27, 2019: 25, 2020: 24, 2021: 25, 2022: 24, 2023: 24, 2024: 23, 2025: 23},
        "score_base": 57.5,
    },
    {
        "id": "NIRF_024",
        "canonical_name": "Aligarh Muslim University",
        "state": "Uttar Pradesh",
        "city": "Aligarh",
        "aliases": ["AMU"],
        "base_rank": 24,
        "rank_trajectory": {2017: 17, 2018: 18, 2019: 21, 2020: 23, 2021: 23, 2022: 25, 2023: 25, 2024: 25, 2025: 25},
        "score_base": 56.4,
    },
    {
        "id": "NIRF_025",
        "canonical_name": "Delhi University",
        "state": "Delhi",
        "city": "New Delhi",
        "aliases": ["DU", "Univ of Delhi", "University of Delhi"],
        "base_rank": 25,
        "rank_trajectory": {2017: 16, 2018: 22, 2019: 24, 2020: 25, 2021: 26, 2022: 26, 2023: 26, 2024: 26, 2025: 26},
        "score_base": 55.8,
    },
    {
        "id": "NIRF_026",
        "canonical_name": "Indian Institute of Management Ahmedabad",
        "state": "Gujarat",
        "city": "Ahmedabad",
        "aliases": ["IIM Ahmedabad", "IIMA", "IIM-A"],
        "base_rank": 26,
        "rank_trajectory": {2017: 26, 2018: 25, 2019: 26, 2020: 27, 2021: 27, 2022: 27, 2023: 27, 2024: 27, 2025: 27},
        "score_base": 55.0,
    },
    {
        "id": "NIRF_027",
        "canonical_name": "Indian Institute of Management Bangalore",
        "state": "Karnataka",
        "city": "Bengaluru",
        "aliases": ["IIM Bangalore", "IIMB", "IIM-B"],
        "base_rank": 27,
        "rank_trajectory": {2017: 29, 2018: 28, 2019: 27, 2020: 28, 2021: 28, 2022: 28, 2023: 28, 2024: 28, 2025: 28},
        "score_base": 54.2,
    },
    {
        "id": "NIRF_028",
        "canonical_name": "National Institute of Technology Rourkela",
        "state": "Odisha",
        "city": "Rourkela",
        "aliases": ["NIT Rourkela", "NITR"],
        "base_rank": 28,
        "rank_trajectory": {2017: 32, 2018: 30, 2019: 29, 2020: 29, 2021: 29, 2022: 29, 2023: 29, 2024: 29, 2025: 29},
        "score_base": 53.6,
    },
    {
        "id": "NIRF_029",
        "canonical_name": "PSG College of Technology",
        "state": "Tamil Nadu",
        "city": "Coimbatore",
        "aliases": ["PSG Tech", "PSG"],
        "base_rank": 29,
        "rank_trajectory": {2017: 40, 2018: 38, 2019: 35, 2020: 33, 2021: 32, 2022: 31, 2023: 30, 2024: 30, 2025: 30},
        "score_base": 52.8,
    },
    {
        "id": "NIRF_030",
        "canonical_name": "SRM Institute of Science and Technology",
        "state": "Tamil Nadu",
        "city": "Chennai",
        "aliases": ["SRM", "SRM University"],
        "base_rank": 30,
        "rank_trajectory": {2017: 45, 2018: 41, 2019: 38, 2020: 36, 2021: 34, 2022: 32, 2023: 31, 2024: 31, 2025: 31},
        "score_base": 52.0,
    },
]

YEARS = list(range(2017, 2026))
CATEGORIES = ["Overall", "Engineering", "University", "Management"]


def generate_dummy_data():
    """Builds realistic dummy NIRF dataset and institutions mapping."""
    rows = []
    inst_records = []

    for inst in INSTITUTIONS:
        inst_records.append({
            "institution_id": inst["id"],
            "canonical_name": inst["canonical_name"],
            "state": inst["state"],
            "aliases": inst["aliases"],
        })

        base_score = inst["score_base"]
        rank_traj = inst["rank_trajectory"]

        for category in CATEGORIES:
            # Shift category rank/score slightly for variety
            cat_score_offset = 0.0
            cat_rank_multiplier = 1.0
            if category == "Engineering":
                cat_score_offset = +2.5 if "Technology" in inst["canonical_name"] or "IIT" in inst["canonical_name"] else -3.0
                cat_rank_multiplier = 0.9 if "Technology" in inst["canonical_name"] or "IIT" in inst["canonical_name"] else 1.3
            elif category == "University":
                cat_score_offset = +3.0 if "University" in inst["canonical_name"] or "Science" in inst["canonical_name"] else -2.0
                cat_rank_multiplier = 0.85 if "University" in inst["canonical_name"] or "Science" in inst["canonical_name"] else 1.2
            elif category == "Management":
                cat_score_offset = +15.0 if "Management" in inst["canonical_name"] else -12.0
                cat_rank_multiplier = 0.5 if "Management" in inst["canonical_name"] else 2.0

            for yr in YEARS:
                # Rank
                base_yr_rank = rank_traj.get(yr, inst["base_rank"])
                yr_rank = max(1, int(round(base_yr_rank * cat_rank_multiplier)))

                # Banding for higher ranks
                if yr_rank > 100:
                    rank_raw = "101-150"
                    final_rank = 101.0
                else:
                    rank_raw = str(yr_rank)
                    final_rank = float(yr_rank)

                # Score
                # Gradual progression over years + small noise
                yr_progress = (yr - 2017) * 0.4
                noise = np.random.normal(0, 0.4)
                final_score = round(max(30.0, min(99.0, base_score + cat_score_offset + yr_progress + noise)), 2)

                # 5 NIRF Parameters (TLR, RPC, GO, OI, PR)
                # Parameters roughly track score with realistic domain variances
                tlr = round(max(20.0, min(100.0, final_score + np.random.normal(2.0, 3.5))), 2)
                rpc = round(max(15.0, min(100.0, final_score + np.random.normal(0.0, 5.0))), 2)
                go = round(max(30.0, min(100.0, final_score + np.random.normal(1.5, 3.0))), 2)
                oi = round(max(35.0, min(100.0, final_score - np.random.normal(5.0, 4.0))), 2)
                pr = round(max(10.0, min(100.0, final_score + np.random.normal(-1.0, 6.0))), 2)

                rows.append({
                    "institution_id": inst["id"],
                    "name": inst["canonical_name"],
                    "state": inst["state"],
                    "city": inst["city"],
                    "year": yr,
                    "category": category,
                    "rank": final_rank,
                    "rank_raw": rank_raw,
                    "score": final_score,
                    "TLR": tlr,
                    "RPC": rpc,
                    "GO": go,
                    "OI": oi,
                    "PR": pr,
                })

    df_nirf = pd.DataFrame(rows)
    df_inst = pd.DataFrame(inst_records)

    # Save to parquet
    nirf_parquet_path = os.path.join(DATA_DIR, "nirf_long.parquet")
    inst_parquet_path = os.path.join(DATA_DIR, "institutions.parquet")
    nirf_csv_path = os.path.join(DATA_DIR, "nirf_long.csv")

    df_nirf.to_parquet(nirf_parquet_path, index=False)
    df_inst.to_parquet(inst_parquet_path, index=False)
    df_nirf.to_csv(nirf_csv_path, index=False)

    print(f"Generated {len(df_nirf)} rows of NIRF dummy data across {len(INSTITUTIONS)} institutions.")
    print(f"Saved: {nirf_parquet_path}")
    print(f"Saved: {inst_parquet_path}")
    print(f"Saved: {nirf_csv_path}")
    return df_nirf, df_inst


if __name__ == "__main__":
    generate_dummy_data()

