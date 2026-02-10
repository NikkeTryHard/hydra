# Data Sources & Datasets (Archived)

> Archived from ECOSYSTEM.md — training data is already converted and ready. Revisit if we need more data or different formats.

---

## Tenhou Logs

| Source | Games | Format | Access | Notes |
|--------|-------|--------|--------|-------|
| [NikkeTryHard/tenhou-to-mjai](https://github.com/NikkeTryHard/tenhou-to-mjai) (Kaggle) | **3M+** Phoenix games | MJAI `.mjson` (gzip) | Public (GitHub Releases + Kaggle) | **Ready to train** — our own pre-converted dataset |
| [Apricot-S/houou-logs](https://github.com/Apricot-S/houou-logs) | All Phoenix room | mjlog XML → SQLite | Public | **Raw log downloader** — replaces archived MahjongRepository/phoenix-logs |
| [tenhou.net/sc/raw/](https://tenhou.net/sc/raw/) | All Tenhou games | ZIP archives with game IDs | Public (usage restrictions) | **Official source** — IDs only, download each game individually |
| [mthrok/tenhou-log-utils](https://github.com/mthrok/tenhou-log-utils) | N/A | mjlog XML parser + downloader | MIT | **Reference parser** for mjlog format |

## Academic Datasets

| Source | Size | Format | Access | Notes |
|--------|------|--------|--------|-------|
| [pjura/mahjong_board_states](https://huggingface.co/datasets/pjura/mahjong_board_states) | 28GB, 650M records | Parquet (510 features + label) | HuggingFace (CC-BY-4.0) | **Tabular dataset** — predict discarded tile from board state |
| [matas234/riichi-mahjong-ml-dataset](https://github.com/matas234/riichi-mahjong-ml-dataset) | Phoenix room | State/label pairs | Public | **Elite player data** — top 0.1% Tenhou players |
| Agony5757/mahjong offline dataset | ~40K games/batch | .mat format | In repo | **Offline RL dataset** — used in ICLR 2022 paper |

## Mahjong Soul Data

| Tool | What It Does | License |
|------|-------------|---------|
| [MahjongRepository/mahjong_soul_api](https://github.com/MahjongRepository/mahjong_soul_api) | Python API wrapper for Majsoul protobuf API, replay fetching | Unlicensed |
| [Cryolite/mahjongsoul_sniffer](https://github.com/Cryolite/mahjongsoul_sniffer) | Sniff, decode, archive Majsoul API requests (Gold+ rooms) | — |
| [Equim-chan/tensoul](https://github.com/Equim-chan/tensoul) | Convert Majsoul logs → Tenhou format | MIT |
| [ssttkkl/tensoul-py](https://github.com/ssttkkl/tensoul-py) | Python port of tensoul | — |
| [jeff39389327/MajsoulPaipuConvert](https://github.com/jeff39389327/majsoulpaipuconvert) | Download from MajSoul Stats → MJAI | — |

## Log Format Converters

| Converter | From → To | Language | License |
|-----------|-----------|----------|---------|
| NikkeTryHard/tenhou-to-mjai | Tenhou mjlog → MJAI | Rust | — |
| fstqwq/mjlog2mjai | Tenhou mjlog → MJAI JSON | Python | MIT |
| EpicOrange/standard-mjlog-converter | Tenhou/Majsoul/Riichi City → Standard | Python | — |
| Equim-chan/tensoul | Majsoul → Tenhou JSON | JavaScript | MIT |
| cht33/RiichiCity-to-Tenhou-Log-Parser | Riichi City → Tenhou | Python | — |
