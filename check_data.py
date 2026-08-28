"""UTKFace 数据检查脚本

作用：
    验证下载的 UTKFace 数据是否完整，以及文件名里的 年龄/性别/人种 标签能否正确解析。
    这是课程要求的"数据检查"：只验证"拿到预期格式"，不能证明数据没有偏差。

运行：
    python check_data.py

预期看到：
    REAL DATA CHECK PASSED
"""

import re
from collections import Counter
from pathlib import Path

# 数据位置（相对于本脚本）：data/raw/utkface/UTKFace/
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "utkface" / "UTKFace"

# 文件名格式：[年龄]_[性别]_[人种]_[日期时间].jpg[.chip].jpg
# 例如：100_0_0_20170112213500903.jpg.chip.jpg
#  - 年龄：0–116 的整数
#  - 性别：0=男，1=女
#  - 人种：0–4（白/黑/亚洲/印度/其他）
FILENAME_PATTERN = re.compile(r"^(\d+)_([01])_([0-4])_\d+\.jpg(\.chip)?\.jpg$")


def main():
    # 1. 文件夹是否存在
    if not DATA_DIR.is_dir():
        print(f"[失败] 找不到数据文件夹：{DATA_DIR}")
        print("请确认已解压 UTKFace 到 data/raw/utkface/UTKFace/")
        return 1

    jpg_files = [p for p in DATA_DIR.iterdir() if p.suffix == ".jpg"]

    # 2. 数量检查
    print(f"图片总数：{len(jpg_files)}")
    if not 20000 <= len(jpg_files) <= 30000:
        print("[警告] 数量不在预期范围（约 20,000–30,000）")
    else:
        print("[通过] 数量在预期范围内")

    # 3. 从文件名解析标签
    ages, genders, races = [], [], []
    unparsed = []
    for f in jpg_files:
        m = FILENAME_PATTERN.match(f.name)
        if m:
            ages.append(int(m.group(1)))
            genders.append(int(m.group(2)))
            races.append(int(m.group(3)))
        else:
            unparsed.append(f.name)

    print(f"成功解析：{len(ages)} / {len(jpg_files)}")
    print(f"解析失败：{len(unparsed)}")
    if unparsed:
        print("解析失败样例（前 5 个）：")
        for name in unparsed[:5]:
            print("  ", name)

    # 4. 标签范围与分布
    if ages:
        print(f"年龄范围：{min(ages)}–{max(ages)}（预期 0–116）")
        print(f"性别分布（0=男, 1=女）：{dict(Counter(genders))}")
        print(f"人种分布（0–4）：{dict(sorted(Counter(races).items()))}")

    # 5. 按人种分组的年龄统计（只用于偏差分析，不作为结论输出）
    if ages:
        race_names = {0: "白种 White", 1: "黑种 Black", 2: "亚洲 Asian", 3: "印度 Indian", 4: "其他 Others"}
        print("\n按人种的年龄统计（n=数量, 均值, 中位数, 范围）：")
        for race in range(5):
            group = [a for a, r in zip(ages, races) if r == race]
            n = len(group)
            if n == 0:
                print(f"  {race_names[race]}（{race}）: n=0")
                continue
            mean_age = sum(group) / n
            sorted_group = sorted(group)
            median_age = sorted_group[n // 2]
            print(f"  {race_names[race]}（{race}）: n={n:5d}  均值={mean_age:5.1f}  中位数={median_age:3d}  范围={min(group)}–{max(group)}")

    # 6. 结论
    ok = len(jpg_files) >= 20000 and not unparsed
    print("\nREAL DATA CHECK PASSED" if ok else "\nREAL DATA CHECK 有警告，请检查上面的输出")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
