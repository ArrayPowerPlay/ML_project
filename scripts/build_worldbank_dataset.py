"""
build_worldbank_dataset.py

Tải các chỉ số kinh tế xã hội từ World Bank cho TẤT CẢ các quốc gia, năm 2000–2024
và xuất ra file Excel có các cột:
Tên Quốc Gia | Mã Quốc Gia | Năm | <12 chỉ số kinh tế xã hội>

Cách sử dụng (từ thư mục gốc của dự án):
  python scripts/build_worldbank_dataset.py --out data/worldbank_2000_2024.xlsx

Các tham số tùy chọn:
  --start 2000 --end 2024                     # năm bắt đầu và kết thúc
  --csv data/worldbank_2000_2024.csv          # lưu thêm file CSV
  --include-aggregates                        # bao gồm các khu vực tổng hợp (mặc định: loại trừ)
"""

import argparse
import sys
import time
import requests
import pandas as pd
from typing import Dict, List

DEFAULT_START = 2000  # Năm bắt đầu mặc định
DEFAULT_END = 2024    # Năm kết thúc mặc định

# Ánh xạ mã chỉ số World Bank -> nhãn cột (tên tiếng Anh chính thức từ World Bank)
INDICATORS = {
    # Dân số
    "SP.POP.TOTL": "Population, total",
    # Nghèo đói
    "SI.POV.DDAY": "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)",
    # Nhân khẩu học
    "SP.POP.GROW": "Population growth (annual %)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    # Kinh tế
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    # Dịch vụ / Tiếp cận
    "SH.STA.SMSS.ZS": "People using safely managed sanitation services (% of population)",
    "EG.ELC.ACCS.ZS": "Access to electricity (% of population)",
    "SH.H2O.BASW.ZS": "People using at least basic drinking water services (% of population)",
    # Môi trường / Đô thị / Lao động
    "EN.GHG.CO2.PC.CE.AR5": "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
    "EN.POP.SLUM.UR.ZS": "Population living in slums (% of urban population)",
    "SL.TLF.CACT.ZS": "Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)",
}

def get_all_countries(exclude_aggregates: bool = True) -> pd.DataFrame:
    """Tải danh sách các quốc gia từ API World Bank, tùy chọn loại trừ các khu vực tổng hợp."""
    url = "https://api.worldbank.org/v2/country"
    params = {"format": "json", "per_page": 400}
    rows = []  # Danh sách để lưu trữ thông tin quốc gia
    session = requests.Session()  # Tạo phiên HTTP để tái sử dụng kết nối

    # Gọi API lần đầu tiên để lấy thông tin về số trang
    r = session.get(url, params=params, timeout=60)
    r.raise_for_status()  # Kiểm tra lỗi HTTP
    meta, data = r.json()
    pages = meta.get("pages", 1)  # Lấy tổng số trang từ metadata

    # Lặp qua tất cả các trang để lấy đầy đủ dữ liệu
    for p in range(1, pages + 1):
        params["page"] = p
        rr = session.get(url, params=params, timeout=60)
        rr.raise_for_status()
        _, dd = rr.json()  # dd là danh sách các quốc gia trên trang này
        for c in dd:
            region_val = (c.get("region") or {}).get("value")
            # Bỏ qua các khu vực tổng hợp (như World, East Asia & Pacific, etc.)
            if exclude_aggregates and region_val == "Aggregates":
                continue
            rows.append({
                "Country Code": c["id"],  # Mã quốc gia (ví dụ: "VNM" cho Việt Nam)
                "Country Name": c["name"],  # Tên quốc gia
            })
    # Chuyển danh sách thành DataFrame và loại bỏ các hàng trùng lặp
    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        raise RuntimeError("Không thể lấy danh sách quốc gia từ API World Bank.")
    return df

def fetch_indicator(ind_code: str, start: int, end: int, countries_df: pd.DataFrame) -> pd.DataFrame:
    """Tải dữ liệu cho một chỉ số cụ thể từ World Bank cho tất cả quốc gia trong khoảng năm."""
    url = f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}"
    params = {"format": "json", "per_page": 20000, "date": f"{start}:{end}"}  # Lấy 20000 bản ghi/trang
    session = requests.Session()

    # Gọi API lần đầu tiên để lấy thông tin về số trang
    r = session.get(url, params=params, timeout=90)
    r.raise_for_status()
    meta, data = r.json()
    if not isinstance(meta, dict):
        raise RuntimeError(f"Phản hồi API không mong muốn cho chỉ số {ind_code}")
    pages = meta.get("pages", 1)  # Số trang dữ liệu

    recs = []  # Danh sách để lưu các bản ghi dữ liệu
    for page in range(1, pages + 1):
        params["page"] = page
        rr = session.get(url, params=params, timeout=90)
        rr.raise_for_status()
        _, dd = rr.json()  # dd là danh sách các bản ghi trên trang này
        if not dd:
            continue  # Nếu không có dữ liệu, bỏ qua trang này
        for row in dd:
            if not row:
                continue
            cc = row.get("countryiso3code")  # Mã quốc gia ISO 3
            yr = row.get("date")  # Năm dữ liệu
            try:
                yr = int(yr)  # Chuyển đổi năm thành số nguyên
            except Exception:
                continue  # Bỏ qua nếu không thể chuyển đổi năm
            recs.append({
                "Country Code": cc,
                "Country Name": (row.get("country") or {}).get("value"),
                "Year": yr,
                ind_code: row.get("value"),  # Giá trị chỉ số (có thể là None)
            })
    # Chuyển danh sách bản ghi thành DataFrame
    df = pd.DataFrame(recs)
    if df.empty:
        # Trả về DataFrame rỗng nhưng có đúng các cột để tránh lỗi khi merge
        return pd.DataFrame(columns=["Country Code", "Country Name", "Year", ind_code])

    # Giữ chỉ các quốc gia được yêu cầu (loại trừ các khu vực tổng hợp nếu cần)
    df = df[df["Country Code"].isin(countries_df["Country Code"])]
    # Loại bỏ trùng lặp theo (quốc gia, năm), giữ lại bản ghi cuối cùng
    df = df.sort_values(["Country Code", "Year"]).drop_duplicates(["Country Code", "Year"], keep="last")
    return df[["Country Code", "Country Name", "Year", ind_code]]

def build_dataset(start: int, end: int, include_aggregates: bool) -> pd.DataFrame:
    """Xây dựng bộ dữ liệu hoàn chỉnh bằng cách kết hợp dữ liệu từ tất cả các chỉ số."""
    # Lấy danh sách tất cả các quốc gia từ World Bank
    countries = get_all_countries(exclude_aggregates=not include_aggregates)

    # Sử dụng dân số (Population) làm bảng cơ sở vì nó bao gồm hầu hết quốc gia-năm
    base_code = "SP.POP.TOTL"
    if base_code not in INDICATORS:
        raise KeyError("SP.POP.TOTL không có trong dictionary INDICATORS.")
    base_df = fetch_indicator(base_code, start, end, countries)

    # Kết hợp các chỉ số khác với bảng dữ liệu cơ sở bằng left join
    df = base_df.copy()
    for code in INDICATORS:
        if code == base_code:
            continue  # Bỏ qua chỉ số cơ sở vì đã có rồi
        part = fetch_indicator(code, start, end, countries)
        # Left join để giữ lại tất cả dòng từ df, thêm các cột chỉ số mới
        df = df.merge(part[["Country Code", "Year", code]], on=["Country Code", "Year"], how="left")

    # Chuẩn hóa tên quốc gia bằng cách sử dụng danh sách quốc gia tham chiếu
    df = df.merge(countries, on="Country Code", how="left", suffixes=("", "_official"))
    df["Country Name"] = df["Country Name_official"].fillna(df["Country Name"])  # Ưu tiên tên chính thức
    df = df.drop(columns=["Country Name_official"])  # Xóa cột tạm thời

    # Đổi tên các cột chỉ số từ mã sang nhãn tiếng Anh
    rename_map = {code: INDICATORS[code] for code in INDICATORS}
    df = df.rename(columns=rename_map)

    # Sắp xếp lại thứ tự cột: Tên Quốc Gia, Mã Quốc Gia, Năm, rồi các chỉ số
    ordered_cols = ["Country Name", "Country Code", "Year"] + [rename_map[c] for c in INDICATORS.keys()]
    df = df[ordered_cols].sort_values(["Country Code", "Year"]).reset_index(drop=True)
    return df

def main():
    """Hàm chính: phân tích tham số dòng lệnh, xây dựng dataset, và xuất ra file."""
    ap = argparse.ArgumentParser(description="Xây dựng file Excel chứa các chỉ số kinh tế xã hội từ World Bank cho tất cả quốc gia, năm 2000–2024.")
    ap.add_argument("--start", type=int, default=DEFAULT_START, help="Năm bắt đầu (mặc định: 2000)")
    ap.add_argument("--end", type=int, default=DEFAULT_END, help="Năm kết thúc (mặc định: 2024)")
    ap.add_argument("--out", type=str, default="worldbank_2000_2024.xlsx", help="Đường dẫn file Excel đầu ra")
    ap.add_argument("--csv", type=str, default=None, help="Tùy chọn: đường dẫn file CSV đầu ra")
    ap.add_argument("--include-aggregates", action="store_true", help="Bao gồm khu vực tổng hợp (mặc định: loại trừ)")

    args = ap.parse_args()

    # Kiểm tra tính hợp lệ của năm
    if args.start > args.end:
        print("Lỗi: --start phải nhỏ hơn hoặc bằng --end", file=sys.stderr)
        sys.exit(2)

    # Hiển thị thông tin bắt đầu
    print(f"[INFO] Xây dựng dataset cho năm {args.start}–{args.end} (Khu vực tổng hợp: {'CÓ BAO GỒM' if args.include_aggregates else 'LOẠI TRỪ'})")
    # Gọi hàm xây dựng dataset
    df = build_dataset(args.start, args.end, include_aggregates=args.include_aggregates)

    # Xuất dữ liệu ra file Excel (một sheet duy nhất)
    print(f"[INFO] Đang ghi file Excel -> {args.out}")
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="WDI_2000_2024")
        try:
            # Điều chỉnh độ rộng của các cột để dễ đọc
            ws = writer.sheets["WDI_2000_2024"]
            ws.set_column(0, 0, 28)   # Cột Tên Quốc Gia: rộng 28
            ws.set_column(1, 1, 12)   # Cột Mã Quốc Gia: rộng 12
            ws.set_column(2, 2, 8)    # Cột Năm: rộng 8
            # Các cột chỉ số còn lại: rộng 36
            ws.set_column(3, len(df.columns)-1, 36)
        except Exception:
            pass  # Bỏ qua lỗi điều chỉnh độ rộng cột nếu có

    # Xuất dữ liệu ra file CSV nếu được yêu cầu
    if args.csv:
        print(f"[INFO] Đang ghi file CSV -> {args.csv}")
        df.to_csv(args.csv, index=False)

    # Hiển thị thông tin hoàn thành
    print(f"[DONE] Tổng số dòng: {len(df):,} | Tổng số cột: {len(df.columns)}")

if __name__ == "__main__":
    # Điểm vào chương trình
    main()
