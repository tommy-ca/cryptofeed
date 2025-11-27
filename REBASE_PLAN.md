# Rebase Plan: feature/kafka-proto-backend onto origin/next

## Current State Analysis

### Branch Status
- **Current Branch**: feature/kafka-proto-backend
- **Target Branch**: origin/next
- **Common Ancestor**: 277f9181 (docs(phase5): Add comprehensive Phase 5 completion final report)
- **Commits Ahead of next**: 65 commits
- **Commits on next not in feature**: 18 commits

### Key Changes on origin/next (Not in our branch)
1. **Documentation Reorganization** (commits 17fe9bfe - 77124d1c):
   - Moved execution reports to `docs/archive/`
   - Organized analysis docs into subcategories
   - Consolidated user-facing docs in `docs/core/`
   - Updated navigation and index files

2. **Kiro System Updates** (commit 77124d1c):
   - Collapsed market-data-kafka-producer to canonical files
   - Updated `.kiro/` structure and templates

3. **QuixStreams Spec** (commits 2cc17ef5 - d2188c68):
   - New specification for cryptofeed-quixstreams-source
   - Requirements, design, and tasks completed

4. **Data Flow Architecture Spec** (commits 374b0ec0 - 411b05f6):
   - New cryptofeed-data-flow-architecture specification

## Potential Conflicts

### High Risk Files (Both branches modified)
1. **.kiro/specs/market-data-kafka-producer/spec.json**
   - origin/next: Collapsed to canonical format
   - feature: Updated with Tasks 20-23 completion status
   - **Resolution Strategy**: Keep our version (more recent task updates)

2. **.kiro/specs/market-data-kafka-producer/tasks.md**
   - origin/next: May have canonical format changes
   - feature: Updated with Tasks 19.2-23 implementation details
   - **Resolution Strategy**: Keep our version (has actual task execution data)

3. **.kiro/** directory structure
   - origin/next: Updated settings, templates, commands
   - feature: May have older structure
   - **Resolution Strategy**: Accept theirs (newer Kiro system structure)

4. **CLAUDE.md**
   - origin/next: May have spec status updates
   - feature: May have different spec status
   - **Resolution Strategy**: Manual merge (combine both updates)

5. **docs/specs/SPEC_STATUS.md**
   - origin/next: Added quixstreams and data-flow specs
   - feature: Updated kafka-producer status
   - **Resolution Strategy**: Manual merge (combine both)

### Medium Risk Files
6. **.env.production.template**
   - feature: Created this file (Task 19.2)
   - origin/next: Might not have it or have different version
   - **Resolution Strategy**: Keep ours (critical for Phase 5)

7. **README.md, docs/README.md**
   - origin/next: Documentation reorganization updates
   - feature: Might have different updates
   - **Resolution Strategy**: Manual review

### Low Risk Areas
8. **Implementation Code** (scripts/, tests/, cryptofeed/):
   - No changes on origin/next
   - All changes are ours
   - **Resolution Strategy**: Automatic (no conflicts expected)

## Rebase Execution Plan

### Phase 1: Backup & Preparation
```bash
# Create backup branch
git branch backup/pre-rebase-$(date +%Y%m%d-%H%M%S)

# Verify clean working tree
git status

# Fetch latest
git fetch origin next
```

### Phase 2: Interactive Rebase
```bash
# Start interactive rebase to handle conflicts carefully
git rebase -i origin/next
```

**Expected Actions**:
- First few commits will likely conflict on `.kiro/` files
- Pause at each conflict for manual resolution
- Use `git status` to see conflicted files
- Resolve conflicts favoring strategies above
- Continue with `git rebase --continue`

### Phase 3: Conflict Resolution Strategy

For each conflict:

1. **Kiro system files** (.kiro/settings/, .kiro/commands/, .kiro/agents/):
   ```bash
   # Accept their version (newer Kiro system)
   git checkout --theirs <file>
   git add <file>
   ```

2. **Spec files** (.kiro/specs/market-data-kafka-producer/):
   ```bash
   # Keep our version (has execution data)
   git checkout --ours <file>
   git add <file>
   ```

3. **CLAUDE.md, SPEC_STATUS.md**:
   ```bash
   # Manual merge required
   # Edit file to combine both changes
   git add <file>
   ```

4. **Implementation files** (scripts/, tests/, cryptofeed/):
   ```bash
   # Should be automatic, but if conflict:
   git checkout --ours <file>
   git add <file>
   ```

### Phase 4: Verification
```bash
# After rebase completes
git log --oneline -10

# Verify no lost commits
git log origin/next..HEAD --oneline | wc -l  # Should still be 65

# Run tests
python -m pytest tests/unit/kafka/ -q --tb=no

# Check git status
git status
```

### Phase 5: Force Push
```bash
# Update remote (force push required after rebase)
git push origin feature/kafka-proto-backend --force-with-lease

# Verify PR updated
gh pr view 16
```

## Rollback Plan

If rebase fails catastrophically:

```bash
# Abort rebase
git rebase --abort

# Or restore from backup
git reset --hard backup/pre-rebase-<timestamp>

# Force push to restore remote
git push origin feature/kafka-proto-backend --force
```

## Post-Rebase Validation

1. ✅ All 65 commits preserved
2. ✅ No duplicate commits
3. ✅ Tests still passing (748 tests)
4. ✅ PR #16 updated with rebased branch
5. ✅ Commit history linear (no merge commits)
6. ✅ Latest origin/next changes incorporated

## Risk Assessment

- **Conflict Risk**: MEDIUM (expect 5-10 file conflicts in .kiro/ and docs/)
- **Data Loss Risk**: LOW (backup branch created, --force-with-lease prevents overwrites)
- **Test Breakage Risk**: LOW (implementation code unchanged on next)
- **Timeline**: 15-30 minutes (depending on conflict complexity)

## Success Criteria

- [ ] Rebase completed without data loss
- [ ] All conflicts resolved correctly
- [ ] Tests passing (748 tests)
- [ ] PR updated successfully
- [ ] Commit history clean and linear
