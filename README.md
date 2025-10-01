## CodeAI Pakistan — AI-Powered Code Quality, Testing & Docs

### Overview
CodeAI Pakistan is a Flask web app that analyzes single-file Python and Java code to accelerate first‑pass reviews. It scores overall quality, estimates coverage, flags potential bugs, infers time complexity, and can generate suggested fixes, unit tests, and documentation. Results are viewable in the UI and exportable as JSON and PDF reports.

### Why this project
- Manual code reviews and documentation are time-consuming and inconsistent.
- Tests are often missing or delayed under delivery pressure.
- Teams need quick, shareable reports to focus on the highest‑value fixes.

This tool provides immediate, automated feedback so developers can prioritize issues and improve quality faster.

### Key features
- Upload a `.py` or `.java` file and get instant analysis
- Overall quality score and quality level (High/Medium/Low)
- Heuristic time complexity (dominant class + confidence)
- Enhanced bug detection with category breakdown and severity
- Unit test and documentation generation
- Real Python coverage collection for uploaded `.py` files (runs discovered tests)
- Optional static analysis enrichments (Bandit, Radon CC)
- Export JSON and PDF reports
- Optional Gemini AI for deeper analysis and improved metrics

---

## Quickstart

### Prerequisites
- Python 3.9+ (project ships with a `venv` directory; you may reuse or create your own)
- On Windows PowerShell
- Optional: Java JDK for `.java` compile checks (`javac` on PATH)

### Install and run (Windows PowerShell)
```bash
# 1) Activate virtual environment
./venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Configure Gemini for enhanced AI analysis
# Persistently (new terminals):
setx GOOGLE_API_KEY "YOUR_KEY_HERE"
# Or only in current session:
$env:GOOGLE_API_KEY="YOUR_KEY_HERE"
# Optional: pick a model (defaults to models/gemini-1.5-pro)
$env:GEMINI_MODEL="models/gemini-1.5-pro"

# 4) Launch the app
python app.py
```
Then open `http://localhost:5000/`.

---

## Using the app
1. Open the web UI.
2. Upload a `.py` or `.java` source file.
3. Click “Analyze”.
4. Explore results:
   - Circular meter: overall quality score
   - Mini-stats: quality, estimated coverage, quality level, complexity
   - Metrics: review precision, test reliability, docs quality, UX
   - Enhanced analysis: time complexity, bug efficiency and severity
   - If Gemini is enabled: quality, maintainability, readability, best practices
5. Review Bug Report and AI‑Corrected Code.
6. Download the JSON or PDF report.

Uploads are stored in `uploads/` with generated `report_*.json` and optional `*.pdf` files.

---

## API Endpoints

### GET `/`
Renders the upload UI.

### POST `/analyze`
Accepts `multipart/form-data` with field `file` (a single `.py` or `.java`). Returns JSON:
```json
{
  "report": {
    "filename": "example.py",
    "language": "Python",
    "timestamp": "2025-01-01T00:00:00Z",
    "scores": {
      "overall": 0,
      "estimated_coverage": 0,
      "complexity": 0,
      "review_score": 0,
      "test_score": 0,
      "doc_score": 0,
      "ux_score": 0,
      "time_complexity": { "dominant": "O(n)", "confidence": 75 },
      "bug_analysis": {
        "detection_efficiency": 0,
        "severity": "Low",
        "total_bugs": 0,
        "bug_categories": { }
      },
      "gemini_quality_score": 0
    },
    "quality_level": "Medium",
    "targets": { "code_quality_target": 80, "coverage_target": 70 },
    "pass_fail": { "code_quality_pass": false, "coverage_pass": false },
    "review": "...",
    "tests": "...",
    "docs": "...",
    "bug_report": "...",
    "corrected_code": "..."
  },
  "report_file": "report_xxx.json",
  "pdf_file": "report_xxx.pdf"
}
```

### GET `/download/<report_name>`
Downloads a previously generated JSON report from `uploads/`.

### GET `/download_pdf/<pdf_name>`
Downloads a generated PDF report (if available) from `uploads/`.

---

## Configuration
Environment variables (load from `apikey.env` or your shell):
- `GOOGLE_API_KEY`: enables Gemini-powered enhanced analysis
- `GEMINI_MODEL` (default: `models/gemini-1.5-pro`)
- `BUG_MODEL_ID` (default: `mrm8488/codebert-base-finetuned-bug-detection`)

If `GOOGLE_API_KEY` is not set, the app falls back to the local Hugging Face pipeline for docs/analysis.

---

## How metrics are computed
- **Overall Quality (0–100)**: Weighted blend of Review, Tests, Docs, UX signals.
- **Estimated Coverage (%)**: Heuristic from test strength; for Python files, actual coverage is collected by executing discovered tests under `coverage.py` in a temp dir.
- **Quality Level**: Derived from overall, structure score, complexity, and coverage.
- **Time Complexity**: Heuristic dominant class among `O(1)`, `O(n)`, `O(n²)`, or `O(log n)` with confidence.
- **Bug Detection**: Combines AI review signals and language patterns; reports detection efficiency, severity (Low/Medium/High), and category counts (syntax, logic, runtime, security, performance). If a classifier model is available, its probability drives severity.
- **Gemini Scores (optional)**: When enabled, Gemini can provide quality/maintainability/readability/best‑practices scores, which can refine the overall score and quality level.

Targets shown in the UI:
- Quality score ≥ 80
- Estimated coverage ≥ 70%

---

## Examples

### cURL
```bash
curl -X POST http://localhost:5000/analyze \
  -F "file=@uploads/qno1.py" \
  -H "Accept: application/json"
```

### Troubleshooting
- `javac not found`: Install a JDK and ensure `javac` is on your PATH for Java compile checks.
- Bandit/Radon unavailable: They are optional; results will omit those enrichments if the tools are missing.
- Large files: The app truncates very long inputs for prompting to keep responses fast.

---

## Tech stack
- Flask
- Hugging Face Transformers (`Salesforce/codet5-base`) for text generation
- Optional: Google Gemini for enhanced analysis
- coverage.py for Python execution coverage
- reportlab for PDF report generation
- Optional: Bandit (security), Radon CC (complexity ranking)

---

## Security & limitations
- Designed for single-file analysis; multi-module projects are out of scope.
- Heuristic metrics are guidance, not ground truth.
- Uploaded files are processed locally and saved to `uploads/` for report generation; clear the directory if needed.



