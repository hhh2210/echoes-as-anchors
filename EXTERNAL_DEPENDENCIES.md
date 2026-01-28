# Code Dependencies and External Repositories

## MI-PEAKS Framework

The Echoic Prompting (EP) experiments are primarily conducted using the MI-PEAKS framework for fair comparison with baseline methods.
https://github.com/ChnQ/MI-Peaks
### Repository Information
- **Location**: External repository (not included in submission)
- **Purpose**: Provides standardized evaluation infrastructure for multi-step reasoning
- **Used for**: Echoic Prompting experiments (Section 4.2 of paper)

### Key Files Referenced
- `MI-Peaks/src/applications/repeat.py` - Main EP implementation
- `MI-Peaks/src/scripts/run_repeat_multi_budget.sh` - Evaluation script

### Note for Reviewers
While the MI-PEAKS implementation is our primary EP evaluation method (for consistency with baselines), we also provide a standalone implementation in `src/evaluation/two_stage_eval.py` that can be run independently without external dependencies.

## How to Access MI-PEAKS
The MI-PEAKS framework is a separate codebase. For reviewers who wish to verify the EP implementation:
1. The standalone version in `src/evaluation/two_stage_eval.py` demonstrates the core EP logic
2. The detailed methodology is described in `EP_IMPLEMENTATION_DETAILS.md`
3. Results from MI-PEAKS experiments are included in the paper and supplementary materials

## Other Dependencies
All other code dependencies are included in this submission or specified in `requirements.txt`.