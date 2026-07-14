# Claude Code Context
See `PROJECT_CONTEXT.md` for full project background.

## Claude-Specific Notes
- **Architectural & Mathematical Rigor:** Focus on deep codebase understanding and validating complex mathematical logic, particularly edge cases, boundary conditions (such as zero-expression mappings), epsilon shifts, and scale-invariance.
- **Manuscript Alignment:** Cross-reference Python implementations in `evograd/` with the theoretical formulations and LaTeX derivations in the `Paper/` directory. Ensure code adjustments do not invalidate the reported historical tables or introduce undocumented paper caveats.
- **Code Review:** Act as the primary reviewer for Codex's implementations. Thoroughly audit diffs to ensure all regressions pass and that API compatibility is strictly preserved before approving merges.
- **Environment & Tooling:** Use `uv` for dependency management and test execution. 
- **Version Control:** Structure architectural suggestions, review notes, and pull request formatting to align with GitHub workflows.

## Current Sprint
Auditing the final compatibility fixes implemented by Codex on the `codex/fix-reviewed-issues` branch. Verifying the mathematical consistency of the evolutionary gradient functions against the EvoGrad manuscript, ensuring all 13 audit regressions pass flawlessly.