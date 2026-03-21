from pathlib import Path

import pandas as pd

from tools.data_io import list_excel_sheets, load_tabular_input


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def test_load_tabular_input_normalizes_csv_parquet_and_xlsx(tmp_path):
    source_df = pd.read_csv(FIXTURE_PATH)

    csv_path = tmp_path / "inforce.csv"
    parquet_path = tmp_path / "inforce.parquet"
    xlsx_path = tmp_path / "inforce.xlsx"

    source_df.to_csv(csv_path, index=False)
    source_df.to_parquet(parquet_path, index=False)
    source_df.to_excel(xlsx_path, index=False)

    for path in (csv_path, parquet_path, xlsx_path):
        loaded = load_tabular_input(str(path))
        assert loaded["Policy_Number"].astype(str).tolist() == source_df["Policy_Number"].astype(str).tolist()
        for column in ["MAC", "MEC", "MAF", "MEF", "MOC"]:
            assert str(loaded[column].dtype) == "float64"


def test_load_tabular_input_uses_requested_excel_sheet(tmp_path):
    workbook_path = tmp_path / "multi_sheet.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {"Policy_Number": ["A1"], "MAC": [0], "MEC": [0.1], "MOC": [1], "MAF": [0], "MEF": [100]}
        ).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame(
            {"Policy_Number": ["B2"], "MAC": [1], "MEC": [0.2], "MOC": [1], "MAF": [500], "MEF": [200]}
        ).to_excel(writer, sheet_name="Second", index=False)

    assert list_excel_sheets(str(workbook_path)) == ["First", "Second"]

    loaded = load_tabular_input(str(workbook_path), sheet_name="Second")
    assert loaded["Policy_Number"].astype(str).tolist() == ["B2"]
