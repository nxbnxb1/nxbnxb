# Enterprise document corpus

Run 37924863485, cleaned on 2026-10-09: documents of TDnet (its robots.txt disallows crawling) and documents
robots.txt disallows were removed. Files are not stored in the repository:
`python scripts/collect_corpus.py fetch corpus/manifest.jsonl --dir corpus_files` downloads them
from the companies' own websites (checksums verified, robots.txt honoured).

| split | language | kind | documents | pages |
|---|---|---|---|---|
| dev | en | digital | 6 | 214 |
| dev | unknown | scan | 6 | 178 |
| dev | vi | digital | 7 | 186 |
| dev | vi | mixed | 1 | 44 |
| test | en | digital | 3 | 198 |
| test | unknown | scan | 19 | 523 |
| test | vi | digital | 17 | 694 |
| test | vi | mixed | 1 | 72 |
| test | vi | scan | 2 | 33 |
| train | en | digital | 9 | 200 |
| train | en | mixed | 1 | 3 |
| train | unknown | scan | 63 | 1273 |
| train | vi | digital | 36 | 787 |
| train | vi | mixed | 13 | 690 |
| train | vi | scan | 9 | 77 |

| country | companies | documents | pages stored |
|---|---|---|---|
| VN | 40 | 193 | 5172 |

193 documents from 40 companies; types: other 35, financial_statement 34, announcement 31, annual_report 27, governance_report 15, charter_regulation 14, explanation_letter 13, resolution 11, shareholder_meeting 7, minutes 6
