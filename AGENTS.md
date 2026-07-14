# Codex Agent Context
See `PROJECT_CONTEXT.md` for full project background.

## Agent-Specific Notes
- **Test Execution:** Always run the full integrated test suite and audit regressions before finalizing changes. Use `uv` for environment and dependency management.
- **Version Control:** Open Pull Requests against the appropriate target branch and follow the template in `.github/pull_request_template.md`. Tag PRs with the relevant GitHub issue number.
- **Manuscript Integrity:** The `Paper/` directory contains LaTeX source files. Do not modify these unless explicitly tasked with updating results tables or paper caveats. 
- **Historical Compatibility:** Ensure any code changes strictly preserve API compatibility and numerical consistency with the historical tables and data reported in the manuscript.
- **Scope of Work:** Focus on targeted execution, bug fixes, and well-scoped component implementations. Flag any required broad architectural or mathematical shifts for human review.

## Current Sprint
Finalizing evidence-oriented reviews for the EvoGrad manuscript. Ensuring all compatibility fixes (e.g., CMA soft selection stability, Problem positional constructor order) pass the 13 audit regressions without altering the reported historical data.