# Farfetch import summary

The locally extracted Kaggle Farfetch dataset was inspected and its JSON schema confirmed. `scripts/import_farfetch.py` normalized all 141 source records successfully, preserving title, description, brand, price, category, attributes, and a valid local primary-image path. The application now prefers the processed catalog. A text index was generated using the documented hash fallback; this is not claimed as semantic embedding retrieval. The suite passed 18 tests.
