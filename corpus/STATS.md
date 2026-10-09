# Enterprise document corpus

Updated by run 37959276808 on 2026-10-09. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured). Long documents are
stored as at most 20 of their pages (scans: 5), the same pages on every fetch.

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 43 | 595 |
| dev | en | mixed | 1 | 13 |
| dev | ja | digital | 161 | 1580 |
| dev | ja | mixed | 11 | 80 |
| dev | ja | scan | 1 | 1 |
| dev | unknown | scan | 17 | 52 |
| dev | vi | digital | 11 | 131 |
| dev | vi | mixed | 1 | 20 |
| test | en | digital | 43 | 497 |
| test | ja | digital | 271 | 2970 |
| test | ja | mixed | 12 | 122 |
| test | ja | scan | 5 | 21 |
| test | unknown | scan | 42 | 141 |
| test | vi | digital | 23 | 270 |
| test | vi | mixed | 1 | 20 |
| test | vi | scan | 3 | 9 |
| train | en | digital | 162 | 2125 |
| train | en | mixed | 3 | 25 |
| train | en | scan | 1 | 5 |
| train | ja | digital | 1047 | 10568 |
| train | ja | mixed | 45 | 722 |
| train | ja | scan | 7 | 11 |
| train | unknown | scan | 115 | 390 |
| train | vi | digital | 60 | 626 |
| train | vi | mixed | 19 | 323 |
| train | vi | scan | 9 | 32 |

| country | companies | documents | pages stored |
|---|---|---|---|
| JP | 370 | 1826 | 18956 |
| VN | 53 | 288 | 2393 |

2114 documents from 423 companies; types: other 676, earnings_summary 320, agm_notice 205, annual_report 147, results_presentation 135, securities_report 120, buyback 90, forecast_revision 80, governance_report 69, dividend 54, sustainability_report 50, financial_statement 47, announcement 37, charter_regulation 20, explanation_letter 18, resolution 16, shareholder_meeting 11, fact_book 10, minutes 9
