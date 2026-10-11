# Enterprise document corpus

Updated by run 38105642880 on 2026-10-11. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured). Long documents are
stored as at most 20 of their pages (scans: 5), the same pages on every fetch.

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 72 | 1017 |
| dev | en | mixed | 4 | 56 |
| dev | en | scan | 1 | 3 |
| dev | ja | digital | 353 | 3539 |
| dev | ja | mixed | 22 | 229 |
| dev | ja | scan | 2 | 2 |
| dev | unknown | digital | 2 | 13 |
| dev | unknown | mixed | 1 | 20 |
| dev | unknown | scan | 41 | 151 |
| dev | vi | digital | 28 | 346 |
| dev | vi | mixed | 3 | 46 |
| dev | vi | scan | 4 | 14 |
| test | en | digital | 86 | 1195 |
| test | en | mixed | 1 | 8 |
| test | en | scan | 1 | 5 |
| test | ja | digital | 613 | 6505 |
| test | ja | mixed | 28 | 389 |
| test | ja | scan | 7 | 31 |
| test | unknown | digital | 2 | 11 |
| test | unknown | mixed | 2 | 40 |
| test | unknown | scan | 95 | 345 |
| test | vi | digital | 36 | 383 |
| test | vi | mixed | 3 | 54 |
| test | vi | scan | 5 | 18 |
| train | en | digital | 378 | 4550 |
| train | en | mixed | 18 | 276 |
| train | en | scan | 6 | 22 |
| train | ja | digital | 2176 | 22494 |
| train | ja | mixed | 112 | 1650 |
| train | ja | scan | 14 | 43 |
| train | unknown | digital | 16 | 126 |
| train | unknown | mixed | 5 | 100 |
| train | unknown | scan | 268 | 961 |
| train | vi | digital | 140 | 1489 |
| train | vi | mixed | 42 | 628 |
| train | vi | scan | 18 | 65 |

| country | companies | documents | pages stored |
|---|---|---|---|
| JP | 1317 | 3851 | 40657 |
| VN | 141 | 754 | 6167 |

4605 documents from 1458 companies; types: other 1513, earnings_summary 665, agm_notice 415, annual_report 357, results_presentation 320, securities_report 246, buyback 168, governance_report 167, forecast_revision 164, sustainability_report 121, financial_statement 114, dividend 94, announcement 77, charter_regulation 50, explanation_letter 39, resolution 35, shareholder_meeting 26, minutes 18, fact_book 16
