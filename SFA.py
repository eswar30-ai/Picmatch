import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill, Font, Alignment
from pathlib import Path
import time


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\ai.testing\Downloads\All India Doctors SFA.xlsx"

OUTPUT_FILE = "duplicate_report2.xlsx"

DUPLICATE_COLUMNS = [
    "DocName",
    "MobileNo",
    "Grade",
    "Category",
    "Qualification",
    "Subarea"
]

start_time = time.time()


# ============================================================
# STEP 1: CHECK FILE
# ============================================================

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

print("=" * 70)
print("DUPLICATE REPORT GENERATION")
print("=" * 70)

print(f"Input file : {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")


# ============================================================
# STEP 2: READ EXCEL
# ============================================================

print("\nReading Excel file...")

df = pd.read_excel(
    INPUT_FILE,
    engine="openpyxl"
)

print(f"Total rows    : {len(df):,}")
print(f"Total columns : {len(df.columns):,}")


# ============================================================
# STEP 3: CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col for col in DUPLICATE_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns found:")

for col in DUPLICATE_COLUMNS:
    print(f"  ✓ {col}")


# ============================================================
# STEP 4: CREATE NORMALIZED VALUES FOR COMPARISON
# ============================================================

print("\nNormalizing duplicate columns...")

df["_DocName"] = (
    df["DocName"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

df["_MobileNo"] = (
    df["MobileNo"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["_Grade"] = (
    df["Grade"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)
df["_Category"] = (
    df["Category"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)
df["_Qualification"] = (
    df["Qualification"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)
df["_Subarea"] = (
    df["Subarea"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# STEP 5: FIND DUPLICATE GROUPS
# ============================================================

print("\nFinding duplicate groups...")

group_columns = [
    "_DocName",
    "_MobileNo",
    "_Grade",
    "Category",
    "Qualification",
    "Subarea"
]

# Count how many records are present in each group
group_counts = (
    df.groupby(group_columns, dropna=False)
      .size()
      .reset_index(name="_duplicate_count")
)

# Keep ONLY groups where count > 1
duplicate_groups = group_counts[
    group_counts["_duplicate_count"] > 1
]

print(
    f"Duplicate groups found: "
    f"{len(duplicate_groups):,}"
)

print(
    f"Total duplicate records "
    f"(including first occurrence): "
    f"{duplicate_groups['_duplicate_count'].sum():,}"
)


# ============================================================
# STEP 6: MERGE DUPLICATE GROUP INFORMATION
# ============================================================

print("\nPreparing duplicate report...")

df_report = df.merge(
    duplicate_groups,
    on=group_columns,
    how="inner"
)


# ============================================================
# STEP 7: CREATE DUPLICATE GROUP ID
# ============================================================

# Create a unique group number for each
# DocName + MobileNo + Grade combination.

df_report["Duplicate_Group"] = (
    df_report
    .groupby(group_columns, sort=False)
    .ngroup() + 1
)


# ============================================================
# STEP 8: MARK FIRST / DUPLICATE OCCURRENCE
# ============================================================

# Within every duplicate group:
#
# First record       -> FIRST OCCURRENCE
# Remaining records  -> DUPLICATE

df_report["Record_Type"] = (
    df_report
    .groupby("Duplicate_Group")
    .cumcount()
    .map(
        lambda x:
        "FIRST OCCURRENCE"
        if x == 0
        else "DUPLICATE"
    )
)


# ============================================================
# STEP 9: REMOVE INTERNAL HELPER COLUMNS
# ============================================================

df_report = df_report.drop(
    columns=[
        "_DocName",
        "_MobileNo",
        "_Grade",
        "_Category",
        "_Qualification",
        "_Subarea",
        "_duplicate_count"
    ]
)


# ============================================================
# STEP 10: REORDER COLUMNS
# ============================================================

# Put group information at the beginning.

original_columns = [
    col for col in df.columns
    if not col.startswith("_")
]

final_columns = [
    "Duplicate_Group",
    "Record_Type"
] + original_columns

df_report = df_report[final_columns]


# ============================================================
# STEP 11: SORT BY DUPLICATE GROUP
# ============================================================

df_report = df_report.sort_values(
    by=["Duplicate_Group"]
)


# ============================================================
# STEP 12: WRITE EXCEL
# ============================================================

print("\nWriting duplicate report...")

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    # Duplicate report
    df_report.to_excel(
        writer,
        sheet_name="Duplicate Report",
        index=False
    )


# ============================================================
# STEP 13: FORMAT EXCEL
# ============================================================

print("Formatting Excel report...")

wb = load_workbook(OUTPUT_FILE)

ws = wb["Duplicate Report"]


# ------------------------------------------------------------
# Header formatting
# ------------------------------------------------------------

header_fill = PatternFill(
    fill_type="solid",
    fgColor="1F4E78"
)

header_font = Font(
    color="FFFFFF",
    bold=True
)

for cell in ws[1]:

    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(
        horizontal="center"
    )


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

first_fill = PatternFill(
    fill_type="solid",
    fgColor="FFF2CC"
)

duplicate_fill = PatternFill(
    fill_type="solid",
    fgColor="FFC7CE"
)

duplicate_font = Font(
    color="9C0006"
)


# ------------------------------------------------------------
# Find columns
# ------------------------------------------------------------

headers = {
    cell.value: cell.column
    for cell in ws[1]
}

group_col = headers["Duplicate_Group"]
type_col = headers["Record_Type"]


# ------------------------------------------------------------
# Apply colors
# ------------------------------------------------------------

for row in range(2, ws.max_row + 1):

    record_type = ws.cell(
        row=row,
        column=type_col
    ).value

    if record_type == "FIRST OCCURRENCE":

        for col in range(1, ws.max_column + 1):

            ws.cell(
                row=row,
                column=col
            ).fill = first_fill

    else:

        for col in range(1, ws.max_column + 1):

            cell = ws.cell(
                row=row,
                column=col
            )

            cell.fill = duplicate_fill
            cell.font = duplicate_font


# ============================================================
# STEP 14: FREEZE HEADER
# ============================================================

ws.freeze_panes = "A2"


# ============================================================
# STEP 15: AUTO FILTER
# ============================================================

ws.auto_filter.ref = ws.dimensions


# ============================================================
# STEP 16: SAVE
# ============================================================

wb.save(OUTPUT_FILE)


# ============================================================
# STEP 17: SUMMARY
# ============================================================

elapsed_time = time.time() - start_time

print("\n" + "=" * 70)
print("DUPLICATE REPORT COMPLETED")
print("=" * 70)

print(f"Original rows              : {len(df):,}")
print(f"Duplicate groups           : {len(duplicate_groups):,}")
print(
    f"Duplicate records          : "
    f"{len(df_report):,}"
)
print(
    f"Output file                : "
    f"{OUTPUT_FILE}"
)
print(
    f"Time taken                 : "
    f"{elapsed_time:.2f} seconds"
)

print("=" * 70)