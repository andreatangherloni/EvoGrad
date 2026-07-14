# Claude Code Context
See `PROJECT_CONTEXT.md` for full project background.

## Claude-Specific Notes
- **Role & Scope:** Rely on Claude for deep architectural analysis, complex multi-file debugging, refactoring, and mathematical verification of evolutionary gradient algorithms (e.g., CMA-ES, L-SHADE, PSO).
- **Environment & Testing:** Use `uv` for all environment management, script execution, and testing (e.g., `uv run pytest`). Always execute the full integrated test suite and audit regressions after structural edits to ensure zero unintended drift in numerical outcomes.
- **Code Review Standards:** When auditing code or reviewing branches from Codex, rigorously check for boundary conditions, edge cases (e.g., constraint feasibility, scalar update paths, seeding reproducibility), and potential side effects across the library.
- **Manuscript & Data Integrity:** Treat the LaTeX files in `Paper/` and historical benchmark tables as immutable ground truth. Do not modify the manuscript unless explicitly directed to document mathematical caveats or update reported figures. 
- **Git Workflow:** Work in dedicated session branches. When preparing code for merge, ensure PRs/MRs adhere to `.gitlab/merge_request_templates/default.md` and document any theoretical or algorithmic trade-offs clearly in the description.

## Current Sprint
Performing final evidence-oriented architectural reviews on the `codex/fix-reviewed-issues` branch. Verifying that recent compatibility fixes (CMA soft selection stability, Problem positional constructor order, and CEC adaptations) are mathematically robust, pass all 13 audit regressions, and preserve exact historical consistency with the EvoGrad manuscript.