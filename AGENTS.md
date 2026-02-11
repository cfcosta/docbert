# CLAUDE.md

## Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Bug Fixing

When I report a bug, don't start by trying to fix it. Instead, start by writing a test that reproduces the bug. Then, have subagents try to fix the bug and prove it with a passing test.

## Tidy up the House

**When the user says "let's tidy up the house", you MUST complete ALL steps below:

1. Review the changes just made and identify simplification opportunities.
2. Apply refactors to:
    - Remove dead code and dead paths.
    - Straighten logic flows.
    - Remove excessive parameters.
    - Remove premature optimization.
3. Run build/tests to verify behavior.
4. Identify optional abstractions or reusable patterns; only suggest them if they clearly improve clarity and keep suggestions brief.

After finishing, you should **land the plane** (check the next section for more details).

## Landing the Plane

**When the user says "let's land the plane"**, you MUST complete ALL steps below. The plane is NOT landed until `jj git push` succeeds. NEVER stop before pushing. NEVER say "ready to push when you are!" - that is a FAILURE.

**MANDATORY WORKFLOW - COMPLETE ALL STEPS:**

1. **File beads issues for any remaining work** that needs follow-up
2. **Ensure all quality gates pass** (only if code changes were made):
   - Run `cargo fmt`, `cargo clippy --all --all-targets`
   - File P0 issues if quality gates are broken
3. **Update beads issues** - close finished work, update status
4. **PUSH TO REMOTE - NON-NEGOTIABLE** - This step is MANDATORY. Execute ALL commands below:

   ```bash
   # Fetch and rebase to catch any remote changes
   jj git fetch
   jj rebase -d main@origin

   # If conflicts in .beads/issues.jsonl, resolve thoughtfully:
   #   - jj restore --from @- .beads/issues.jsonl (accept previous version)
   #   - bd import -i .beads/issues.jsonl (re-import)
   #   - Or manually resolve, then jj resolve --mark .beads/issues.jsonl

   # Sync the database (exports to JSONL, commits)
   bd sync

   # MANDATORY: Push everything to remote
   # DO NOT STOP BEFORE THIS COMMAND COMPLETES
   jj git push

   # MANDATORY: Verify push succeeded
   jj log -r 'main@origin'  # MUST show your latest changes
   ```

   **CRITICAL RULES:**

   - The plane has NOT landed until `jj git push` completes successfully
   - NEVER stop before `jj git push` - that leaves work stranded locally
   - NEVER say "ready to push when you are!" - YOU must push, not the user
   - If `jj git push` fails, resolve the issue and retry until it succeeds
   - The user is managing multiple agents - unpushed work breaks their coordination workflow

5. **Verify clean state** - Ensure all changes are committed AND PUSHED, no untracked files remain
6. **Choose a follow-up issue for next session**
   - Provide a prompt for the user to give to you in the next session
   - Format: "Continue work on bd-X: [issue title]. [Brief context about what's been done and what's next]"

**REMEMBER: Landing the plane means EVERYTHING is pushed to remote. No exceptions. No "ready when you are". PUSH IT.**

**Example "land the plane" session:**

```bash
# 1. File remaining work
bd create "Add integration tests for sync" -t task -p 2 --json

# 2. Run quality gates (only if code changes were made)
cargo fmt
cargo clippy --all --all-targets

# 3. Close finished issues
bd close bd-42 bd-43 --reason "Completed" --json

# 4. PUSH TO REMOTE - MANDATORY, NO STOPPING BEFORE THIS IS DONE
jj git fetch
jj rebase -d main@origin
# If conflicts in .beads/issues.jsonl, resolve thoughtfully:
#   - jj restore --from @- .beads/issues.jsonl (accept previous)
#   - bd import -i .beads/issues.jsonl (re-import)
#   - Or manually resolve, then jj resolve --mark
bd sync           # Export/import/commit
jj git push       # MANDATORY - THE PLANE IS STILL IN THE AIR UNTIL THIS SUCCEEDS
jj log -r 'main@origin'  # MUST verify changes are on remote

# 5. Verify everything is clean and pushed
jj status

# 6. Choose next work
bd ready --json
bd show bd-44 --json
```

**Then provide the user with:**

- Summary of what was completed this session
- What issues were filed for follow-up
- Status of quality gates (all passing / issues filed)
- Confirmation that ALL changes have been pushed to remote
- Recommended prompt for next session

**CRITICAL: Never end a "land the plane" session without successfully pushing. The user is coordinating multiple agents and unpushed work causes severe rebase conflicts.**
