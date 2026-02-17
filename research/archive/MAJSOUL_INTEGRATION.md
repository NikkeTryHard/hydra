# Mahjong Soul Integration (Archived)

> Archived from ECOSYSTEM.md — revisit when training is done and we're ready to deploy.

---

## Safe to Use (MIT Licensed)

| Repo | License | What It Does | Hydra Use |
|------|---------|-------------|-----------|
| [747929791/majsoul_wrapper](https://github.com/747929791/majsoul_wrapper) | MIT | Complete SDK for Majsoul automated play: WebSocket interception, game state callbacks, action execution | **Majsoul bot SDK** — subclass `MajsoulHandler`, implement AI logic |
| [Equim-chan/tensoul](https://github.com/Equim-chan/tensoul) | MIT | Convert Majsoul logs → Tenhou format | **Replay conversion** for training data |
| [SAPikachu/amae-koromo](https://github.com/SAPikachu/amae-koromo) | MIT | Mahjong Soul stats site (牌谱屋), Jade/Throne room tracking | **Benchmark data** — top player performance calibration |
| [zyr17/MajsoulPaipuAnalyzer](https://github.com/zyr17/MajsoulPaipuAnalyzer) | MIT | Replay crawler + statistical analysis (agari rate, deal-in, riichi stats) | **Performance metrics reference** — what stats matter |

## Reference Only (Copyleft — study, don't copy)

| Repo | License | What It Does | Study For |
|------|---------|-------------|-----------|
| [shinkuan/Akagi](https://github.com/shinkuan/Akagi) | AGPL-3.0 + Commons Clause | MITM AI assistant, Majsoul→MJAI bridge, AutoPlay | Protocol bridge architecture |
| [latorc/MahjongCopilot](https://github.com/latorc/MahjongCopilot) | GPL-3.0 | Mortal-based copilot, Playwright-embedded Chromium, in-game HUD | Playwright integration pattern |
| [Xe-Persistent/Akagi-NG](https://github.com/Xe-Persistent/Akagi-NG) | AGPL-3.0 | Next-gen Akagi rewrite, Electron UI, Desktop Mode (zero-config embedded browser) | Desktop Mode pattern |

## ToS Risk Assessment

| Activity | Risk | Notes |
|----------|------|-------|
| MITM traffic interception | 🔴 HIGH | Active bans reported (Oct 2024) |
| Automated play (AutoPlay) | 🔴 VERY HIGH | Pattern detection via play speed/timing |
| Replay data fetching via API | 🟡 MEDIUM | Official APIs but scale may trigger flags |
| Stats tracking (amae-koromo style) | 🟢 LOW | Widely used, no reported bans |
| Offline replay analysis | 🟢 LOW | No interaction with live game |

## Mahjong Soul Data Tools

> See [archive/DATA_SOURCES.md § Mahjong Soul Data](DATA_SOURCES.md#mahjong-soul-data) for data extraction tools (mahjong_soul_api, mahjongsoul_sniffer, tensoul-py, MajsoulPaipuConvert).
