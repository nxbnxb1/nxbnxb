# Enterprise document corpus

Updated by run 38018089142 on 2026-10-10. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured). Long documents are
stored as at most 20 of their pages (scans: 5), the same pages on every fetch.

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 69 | 961 |
| dev | en | mixed | 4 | 56 |
| dev | en | scan | 1 | 3 |
| dev | ja | digital | 337 | 3391 |
| dev | ja | mixed | 18 | 170 |
| dev | ja | scan | 2 | 2 |
| dev | unknown | scan | 38 | 136 |
| dev | vi | digital | 28 | 338 |
| dev | vi | mixed | 3 | 46 |
| dev | vi | scan | 2 | 4 |
| test | en | digital | 81 | 1095 |
| test | en | mixed | 1 | 8 |
| test | en | scan | 1 | 5 |
| test | ja | digital | 578 | 6199 |
| test | ja | mixed | 25 | 335 |
| test | ja | scan | 7 | 31 |
| test | unknown | scan | 85 | 305 |
| test | vi | digital | 36 | 373 |
| test | vi | mixed | 5 | 94 |
| test | vi | scan | 8 | 33 |
| train | en | digital | 346 | 4211 |
| train | en | mixed | 14 | 208 |
| train | en | scan | 5 | 17 |
| train | ja | digital | 2053 | 21135 |
| train | ja | mixed | 106 | 1571 |
| train | ja | scan | 14 | 43 |
| train | unknown | scan | 233 | 812 |
| train | vi | digital | 143 | 1470 |
| train | vi | mixed | 38 | 600 |
| train | vi | scan | 29 | 121 |

| country | companies | documents | pages stored |
|---|---|---|---|
| JP | 1289 | 3631 | 38243 |
| VN | 104 | 679 | 5530 |

4310 documents from 1393 companies; types: other 1432, earnings_summary 625, agm_notice 389, annual_report 328, results_presentation 302, securities_report 234, buyback 158, governance_report 153, forecast_revision 152, sustainability_report 112, financial_statement 96, dividend 91, announcement 74, charter_regulation 43, explanation_letter 34, resolution 32, shareholder_meeting 22, minutes 18, fact_book 15
