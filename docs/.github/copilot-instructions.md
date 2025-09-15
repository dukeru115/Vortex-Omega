# Copilot Onboarding Instructions for Vortex-Omega

<Goals>
- Reduce the likelihood of PRs getting rejected due to build failures, validation pipeline failures, or misbehavior.
- Minimize bash command and build failures.
- Allow the agent to complete tasks faster by reducing exploration and trial-and-error.
</Goals>

<Limitations>
- Instructions must not exceed 2 pages.
- Instructions must not be task-specific.
</Limitations>

<HighLevelDetails>
- Repository Purpose: Hybrid AI system repository for token-level ESC patterns, multi-agent consensus (Kuramoto/ADMM), causal world models (RT-2, Dreamer), interpretable outputs (Integrated Gradients, ESC telemetry), and reproducible CI notebooks.
- Project Type: AI/ML framework with multi-language scripts.
- Languages & Frameworks: Python 3.8+, Jupyter notebooks, Docker, Bash scripts.
- Target Runtimes: Linux, Docker containers, cross-platform Python environment.
- Key Components:
  - src/: core model and algorithm implementations.
  - scripts/: execution, setup, and automation scripts.
  - notebooks/: demonstrations and experiments.
  - tests/: unit and integration tests.
  - docs/: documentation and guides.
  - config/: configuration files and parameters.
  - venv/: optional Python virtual environment.
</HighLevelDetails>

<BuildInstructions>
1. Environment Setup:
   - Always clean environment before starting.
   - Create virtual environment:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. Build / Bootstrap:
   - Always install dependencies before building.
   - Build using Makefile or scripts:
     ```bash
     make
     ```
   - Docker build (optional):
     ```bash
     docker build -t vortex-omega .
     ```

3. Run Tests:
   - Run all tests:
     ```bash
     pytest tests/
     ```
   - Run in Docker if using containerized environment:
     ```bash
     docker run --rm vortex-omega pytest tests/
     ```

4. Run Model / Scripts:
   ```bash
   python scripts/run_model.py
   docker run --rm vortex-omega python scripts/run_model.py

5. CI/CD:

Follow .gitlab-ci.yml or GitHub workflows.

Always ensure tests pass and build succeeds before merging.



6. Optional / Hidden Setup:

Document any required environment variables, runtime versions, or auxiliary services.

Log any errors and workarounds. </BuildInstructions>




<ProjectLayout>
- Major directories:
  - src/, scripts/, tests/, notebooks/, docs/, config/, venv/
- Key files:
  - README.md (update whenever working in a folder)
  - CONTRIBUTING.md
  - Makefile, Dockerfile, requirements.txt
  - .gitlab-ci.yml or GitHub workflows
  - Main entry points in scripts/run_model.py
- Validation steps:
  - Run tests
  - Ensure build succeeds
  - Validate Docker container execution (if applicable)
- Dependencies not obvious from layout:
  - Python 3.8+, Docker, Bash
- Additional notes:
  - Always update README.md in every folder where changes are made
  - Maintain a revision history with changes and reasons
  - Check for HACK, TODO, or workaround comments in code
</ProjectLayout><StepsToFollow>
- Think as top 0.01% project manager:
  - Analyze repository structure deeply.
  - Identify dependencies and potential problems before making changes.
  - Solve issues step-by-step; do not skip validation.
- Update all README.md files in folders where changes occur.
- Write code and documentation in English.
- Track revision history for every change.
- Always run tests and build after each significant change.
- Trust these instructions; only search code if instructions are incomplete or wrong.
</StepsToFollow>


