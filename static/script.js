// Performance-optimized DOM ready handler
document.addEventListener("DOMContentLoaded", () => {
    // Cache DOM elements for better performance
    const elements = {
        fileInput: document.getElementById("fileInput"),
        fileLabel: document.getElementById("fileLabel"),
        uploadForm: document.getElementById("uploadForm"),
        analyzeBtn: document.getElementById("analyzeBtn"),
        resetBtn: document.getElementById("resetBtn"),
        errorMsg: document.getElementById("errorMsg"),
        gear: document.getElementById("gear"),
        circleFill: document.getElementById("circleFill"),
        circleText: document.getElementById("circleText"),
        qualityStat: document.getElementById("qualityStat"),
        coverageStat: document.getElementById("coverageStat"),
        resultsArea: document.getElementById("results"),
        bugReportPre: document.getElementById("bugReportPre"),
        correctedCodePre: document.getElementById("correctedCodePre"),
        barReview: document.getElementById("barReview"),
        barTest: document.getElementById("barTest"),
        barDoc: document.getElementById("barDoc"),
        barUx: document.getElementById("barUx"),
        numReview: document.getElementById("numReview"),
        numTest: document.getElementById("numTest"),
        numDoc: document.getElementById("numDoc"),
        numUx: document.getElementById("numUx"),
        downloadReport: document.getElementById("downloadReport"),
        downloadPdf: document.getElementById("downloadPdf"),
        geminiMetrics: document.getElementById("geminiMetrics"),
        qualityLevel: document.getElementById("qualityLevel"),
        timeComplexity: document.getElementById("timeComplexity")
    };

    // Validate required elements
    const requiredElements = ['fileInput', 'fileLabel', 'uploadForm', 'analyzeBtn', 'resetBtn'];
    const missingElements = requiredElements.filter(id => !elements[id]);
    if (missingElements.length > 0) {
        console.error('Missing required elements:', missingElements);
        return;
    }

    // Utility functions
    const safeSetText = (element, text) => {
        if (element) element.textContent = text;
    };

    const safeSetDisplay = (element, display) => {
        if (element) element.style.display = display;
    };

    const safeSetAriaHidden = (element, hidden) => {
        if (element) element.setAttribute('aria-hidden', hidden);
    };

    const showError = (message) => {
        if (elements.errorMsg) {
            elements.errorMsg.textContent = message;
            elements.errorMsg.style.display = "block";
            elements.errorMsg.setAttribute('aria-hidden', 'false');
            elements.errorMsg.focus();
        }
    };

    const hideError = () => {
        if (elements.errorMsg) {
            elements.errorMsg.style.display = "none";
            elements.errorMsg.setAttribute('aria-hidden', 'true');
        }
    };

    const setLoadingState = (loading) => {
        if (elements.analyzeBtn) {
            elements.analyzeBtn.disabled = loading;
            elements.analyzeBtn.classList.toggle('loading', loading);
            
            const btnText = elements.analyzeBtn.querySelector('.btn-text');
            const btnLoading = elements.analyzeBtn.querySelector('.btn-loading');
            
            if (btnText && btnLoading) {
                btnText.style.display = loading ? 'none' : 'inline';
                btnLoading.style.display = loading ? 'inline' : 'none';
            }
        }
    };

    // File input change handler
    elements.fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            safeSetText(elements.fileLabel, file.name);
            hideError();
        } else {
            safeSetText(elements.fileLabel, "Choose a source file…");
        }
    });

    // Reset button handler
    elements.resetBtn.addEventListener("click", () => {
        elements.fileInput.value = "";
        safeSetText(elements.fileLabel, "Choose a source file…");
        safeSetDisplay(elements.resultsArea, "none");
        safeSetDisplay(elements.geminiMetrics, "none");
        hideError();
        safeSetText(elements.circleText, "--%");
        if (elements.circleFill) {
            elements.circleFill.style.background = "conic-gradient(var(--accent) 0deg, rgba(255,255,255,0.06) 0deg)";
        }
        safeSetText(elements.qualityStat, "—");
        safeSetText(elements.coverageStat, "—");
        safeSetText(elements.qualityLevel, "—");
        safeSetText(elements.timeComplexity, "—");
        
        // Reset button states
        setLoadingState(false);
    });

    // Form submission handler
    elements.uploadForm.addEventListener("submit", (ev) => {
        ev.preventDefault();
        hideError();
        
        if (!elements.fileInput.files.length) {
            showError("Please select a file first.");
            return;
        }

        const file = elements.fileInput.files[0];
        const maxSize = 10 * 1024 * 1024; // 10MB limit
        
        if (file.size > maxSize) {
            showError("File size too large. Please select a file smaller than 10MB.");
                    return;
        }

        const formData = new FormData();
        formData.append("file", file);

        setLoadingState(true);
        hideError();

        // Show progress indicator and animate gear faster
        if (elements.gear) {
            elements.gear.style.display = "block";
            elements.gear.style.animationDuration = "1.8s";
        }

        fetch("/analyze", { 
            method: "POST", 
            body: formData 
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(payload => {
            if (payload.error) {
                throw new Error(payload.error);
                }

                const report = payload.report;

                // animate the circular meter and fill
                const overall = report.scores.overall;
                const coverage = report.scores.estimated_coverage;

                // Update circle fill via conic gradient (smooth)
                const deg = Math.round((overall / 100) * 360);
            if (elements.circleFill) {
                elements.circleFill.style.background = `conic-gradient(var(--accent) ${deg}deg, rgba(255,255,255,0.06) ${deg}deg)`;
            }
            safeSetText(elements.circleText, `${overall}%`);
            safeSetText(elements.qualityStat, `${overall}%`);
            safeSetText(elements.coverageStat, `${coverage}%`);
                
            // Enhanced metrics (robust defaults + logging)
            const qualityLevel = elements.qualityLevel;
            const timeComplexity = elements.timeComplexity;
            const qualityLevelValue = document.getElementById('qualityLevelValue');
                const timeComplexityValue = document.getElementById('timeComplexityValue');
                const bugEfficiencyValue = document.getElementById('bugEfficiencyValue');
                const bugSeverityValue = document.getElementById('bugSeverityValue');

                const s = report.scores || {};
                console.debug('Scores:', s);

                const setText = (el, val) => { if (el) el.textContent = val; };

                setText(qualityLevel, s.quality_level || 'N/A');
                setText(qualityLevelValue, s.quality_level || 'N/A');

                const dominant = s.time_complexity && s.time_complexity.dominant;
                setText(timeComplexity, dominant || 'N/A');
                setText(timeComplexityValue, dominant || 'N/A');

                const eff = s.bug_analysis && s.bug_analysis.detection_efficiency;
                setText(bugEfficiencyValue, (typeof eff === 'number') ? `${eff}%` : 'N/A');
                const sev = s.bug_analysis && s.bug_analysis.severity;
                setText(bugSeverityValue, sev || 'N/A');

                // Handle Gemini enhanced metrics
                const geminiQualityValue = document.getElementById('geminiQualityValue');
                const maintainabilityValue = document.getElementById('maintainabilityValue');
                const readabilityValue = document.getElementById('readabilityValue');
                const bestPracticesValue = document.getElementById('bestPracticesValue');

                if (s.gemini_quality_score || s.maintainability_score || s.readability_score || s.best_practices_score) {
                    if (elements.geminiMetrics) elements.geminiMetrics.style.display = "block";
                    
                    setText(geminiQualityValue, s.gemini_quality_score ? `${s.gemini_quality_score}/100` : 'N/A');
                    setText(maintainabilityValue, s.maintainability_score ? `${s.maintainability_score}/100` : 'N/A');
                    setText(readabilityValue, s.readability_score ? `${s.readability_score}/100` : 'N/A');
                    setText(bestPracticesValue, s.best_practices_score ? `${s.best_practices_score}/100` : 'N/A');
                } else {
                    if (elements.geminiMetrics) elements.geminiMetrics.style.display = "none";
                }

                // show breakdown bars
                const r = report.scores.review_score;
                const t = report.scores.test_score;
                const d = report.scores.doc_score;
                const u = report.scores.ux_score;

                if (elements.barReview) elements.barReview.style.width = `${ r }%`;
                if (elements.barTest) elements.barTest.style.width = `${ t }%`;
                if (elements.barDoc) elements.barDoc.style.width = `${ d }%`;
                if (elements.barUx) elements.barUx.style.width = `${ u }%`;

                if (elements.numReview) elements.numReview.textContent = `${ r }%`;
                if (elements.numTest) elements.numTest.textContent = `${ t }%`;
                if (elements.numDoc) elements.numDoc.textContent = `${ d }%`;
                if (elements.numUx) elements.numUx.textContent = `${ u }%`;

                // display bug report and corrected code (Gemini-backed if available)
                if (elements.bugReportPre) elements.bugReportPre.textContent = report.bug_report || "[no bug report]";
                if (elements.correctedCodePre) elements.correctedCodePre.textContent = report.corrected_code || "[no corrected code]";

                // show report area
                if (elements.resultsArea) elements.resultsArea.style.display = "block";

                // download links
                if (payload.report_file && elements.downloadReport) {
                    elements.downloadReport.href = `/download/${ payload.report_file }`;
                    elements.downloadReport.style.display = "inline-block";
                } else if (elements.downloadReport) {
                    elements.downloadReport.style.display = "none";
                }

                if (payload.pdf_file && elements.downloadPdf) {
                    elements.downloadPdf.href = `/download_pdf/${ payload.pdf_file }`;
                    elements.downloadPdf.style.display = "inline-block";
                } else if (elements.downloadPdf) {
                    elements.downloadPdf.style.display = "none";
                }

                // show pass/fail badges (optional small toast)
                if (!report.pass_fail.code_quality_pass || !report.pass_fail.coverage_pass) {
                    // small visual cue
                    setTimeout(() => {
                        alert("Note: One or more targets not met. See detailed suggestions in the Review Summary.");
                    }, 200);
                }
            })
        .catch(error => {
            console.error("Analysis failed:", error);
            showError(`Analysis failed: ${error.message}`);
        })
        .finally(() => {
            setLoadingState(false);
            if (elements.gear) {
                elements.gear.style.animationDuration = "6s";
                elements.gear.style.display = "none";
            }
            });
    });

});
