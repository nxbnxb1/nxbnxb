# Enterprise document corpus

Updated by run 38111311592 on 2026-10-11. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured). Long documents are
stored as at most 20 of their pages (scans: 5), the same pages on every fetch.

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 56 | 730 |
| dev | en | mixed | 10 | 193 |
| dev | ja | digital | 339 | 3340 |
| dev | ja | mixed | 30 | 352 |
| dev | other | digital | 4 | 28 |
| dev | unknown | scan | 69 | 469 |
| dev | vi | digital | 20 | 238 |
| dev | vi | mixed | 5 | 86 |
| test | en | digital | 74 | 955 |
| test | en | mixed | 7 | 128 |
| test | ja | digital | 585 | 6134 |
| test | ja | mixed | 46 | 600 |
| test | other | digital | 1 | 20 |
| test | unknown | scan | 136 | 810 |
| test | vi | digital | 25 | 263 |
| test | vi | mixed | 5 | 74 |
| train | en | digital | 311 | 3755 |
| train | en | mixed | 25 | 427 |
| train | ja | digital | 2041 | 20821 |
| train | ja | mixed | 183 | 2638 |
| train | other | digital | 6 | 39 |
| train | other | mixed | 7 | 109 |
| train | unknown | scan | 481 | 3011 |
| train | vi | digital | 109 | 1200 |
| train | vi | mixed | 30 | 404 |

| country | companies | documents | pages stored |
|---|---|---|---|
| JP | 1317 | 3851 | 40657 |
| VN | 141 | 754 | 6167 |

4605 documents from 1458 companies; types: other 1513, earnings_summary 665, agm_notice 415, annual_report 357, results_presentation 320, securities_report 246, buyback 168, governance_report 167, forecast_revision 164, sustainability_report 121, financial_statement 114, dividend 94, announcement 77, charter_regulation 50, explanation_letter 39, resolution 35, shareholder_meeting 26, minutes 18, fact_book 16
