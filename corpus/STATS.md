# Enterprise document corpus

Built by run 37924863485 on 2026-10-09. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own publications (checksums verified).

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 6 | 214 |
| dev | ja | digital | 42 | 187 |
| dev | ja | mixed | 4 | 40 |
| dev | unknown | scan | 6 | 178 |
| dev | vi | digital | 7 | 186 |
| dev | vi | mixed | 1 | 44 |
| test | en | digital | 3 | 198 |
| test | ja | digital | 62 | 279 |
| test | ja | mixed | 4 | 10 |
| test | unknown | scan | 21 | 525 |
| test | vi | digital | 17 | 694 |
| test | vi | mixed | 1 | 72 |
| test | vi | scan | 2 | 33 |
| train | en | digital | 10 | 202 |
| train | en | mixed | 1 | 3 |
| train | ja | digital | 262 | 1542 |
| train | ja | mixed | 13 | 193 |
| train | unknown | scan | 66 | 1310 |
| train | vi | digital | 38 | 1010 |
| train | vi | mixed | 13 | 690 |
| train | vi | scan | 14 | 303 |

593 documents from 257 companies; types: other 214, buyback 78, earnings_summary 49, financial_statement 35, announcement 32, annual_report 28, personnel 23, forecast_revision 21, results_presentation 19, financing 16, governance_report 16, charter_regulation 15, explanation_letter 13, resolution 11, shareholder_meeting 7, dividend 6, minutes 6, agm_notice 3, securities_report 1
