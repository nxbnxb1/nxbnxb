# Enterprise document corpus

Updated by run 38110409022 on 2026-10-11. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured). Long documents are
stored as at most 20 of their pages (scans: 5), the same pages on every fetch.

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 65 | 892 |
| dev | en | mixed | 3 | 53 |
| dev | ja | digital | 352 | 3519 |
| dev | ja | mixed | 22 | 229 |
| dev | ja | scan | 2 | 2 |
| dev | other | digital | 12 | 173 |
| dev | other | mixed | 1 | 3 |
| dev | other | scan | 1 | 3 |
| dev | unknown | digital | 2 | 13 |
| dev | unknown | mixed | 1 | 20 |
| dev | unknown | scan | 43 | 161 |
| dev | vi | digital | 24 | 318 |
| dev | vi | mixed | 3 | 46 |
| dev | vi | scan | 2 | 4 |
| test | en | digital | 81 | 1113 |
| test | en | mixed | 1 | 8 |
| test | en | scan | 1 | 5 |
| test | ja | digital | 609 | 6462 |
| test | ja | mixed | 28 | 389 |
| test | ja | scan | 4 | 16 |
| test | other | digital | 10 | 145 |
| test | other | scan | 3 | 15 |
| test | unknown | digital | 2 | 11 |
| test | unknown | mixed | 2 | 40 |
| test | unknown | scan | 95 | 345 |
| test | vi | digital | 35 | 363 |
| test | vi | mixed | 3 | 54 |
| test | vi | scan | 5 | 18 |
| train | en | digital | 333 | 4168 |
| train | en | mixed | 15 | 228 |
| train | en | scan | 2 | 10 |
| train | ja | digital | 2165 | 22402 |
| train | ja | mixed | 108 | 1603 |
| train | ja | scan | 13 | 40 |
| train | other | digital | 69 | 614 |
| train | other | mixed | 6 | 66 |
| train | other | scan | 6 | 20 |
| train | unknown | digital | 16 | 126 |
| train | unknown | mixed | 10 | 200 |
| train | unknown | scan | 267 | 956 |
| train | vi | digital | 127 | 1349 |
| train | vi | mixed | 38 | 557 |
| train | vi | scan | 18 | 65 |

| country | companies | documents | pages stored |
|---|---|---|---|
| JP | 1317 | 3851 | 40657 |
| VN | 141 | 754 | 6167 |

4605 documents from 1458 companies; types: other 1513, earnings_summary 665, agm_notice 415, annual_report 357, results_presentation 320, securities_report 246, buyback 168, governance_report 167, forecast_revision 164, sustainability_report 121, financial_statement 114, dividend 94, announcement 77, charter_regulation 50, explanation_letter 39, resolution 35, shareholder_meeting 26, minutes 18, fact_book 16
