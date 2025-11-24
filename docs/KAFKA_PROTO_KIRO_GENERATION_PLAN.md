# Kafka & Protobuf Code Improvement - Kiro Generation Plan

**Plan for generating design and tasks using `kiro:spec-*` commands**

---

## Overview

This document provides a step-by-step plan for generating the design and tasks documents using Kiro's specification workflow commands.

**Spec Name**: `kafka-proto-code-improvement`  
**Current Status**: Requirements Phase (ready for design generation)

---

## Prerequisites Checklist

Before generating design, ensure:

- [x] ✅ Specification directory exists: `.kiro/specs/kafka-proto-code-improvement/`
- [x] ✅ `spec.json` created with metadata
- [x] ✅ `requirements.md` drafted with functional requirements
- [x] ✅ Steering files available: `.kiro/steering/*.md`
- [x] ✅ Design templates available: `.kiro/settings/templates/specs/design.md`
- [x] ✅ Task templates available: `.kiro/settings/templates/specs/tasks.md`

**Status**: ✅ All prerequisites met

---

## Phase 1: Generate Design

### Step 1.1: Review Requirements

**Action**: Review requirements document for completeness

```bash
# View requirements
cat .kiro/specs/kafka-proto-code-improvement/requirements.md
```

**Review Checklist**:
- [ ] All functional requirements (FR1-FR6) are clear
- [ ] Non-functional requirements (NFR1-NFR3) are defined
- [ ] Scope boundaries are explicit
- [ ] Success criteria are measurable
- [ ] Engineering principles are aligned

**Current Status**: ✅ Requirements ready for review

---

### Step 1.2: Approve Requirements

**Action**: Mark requirements as approved in `spec.json`

**Manual Update** (if needed):
```json
{
  "phases": {
    "requirements": {
      "status": "approved",  // Change from "draft"
      "approved_date": "2025-01-15"
    }
  }
}
```

**Or**: Use `-y` flag to auto-approve during design generation

---

### Step 1.3: Generate Design Document

**Command**:
```bash
kiro:spec-design kafka-proto-code-improvement
```

**Or with auto-approve**:
```bash
kiro:spec-design kafka-proto-code-improvement -y
```

**What This Does**:
1. Validates requirements exist
2. Reads context from:
   - `.kiro/specs/kafka-proto-code-improvement/requirements.md`
   - `.kiro/specs/kafka-proto-code-improvement/spec.json`
   - `.kiro/steering/*.md` (project context)
   - `.kiro/settings/rules/design-*.md` (design rules)
   - `.kiro/settings/templates/specs/design.md` (template)
3. Generates comprehensive design document
4. Updates `spec.json` with design phase status

**Expected Output**:
- Updated `design.md` with:
  - Architecture overview
  - Module design (Kafka + Protobuf)
  - Class hierarchy (`KafkaBackendBase`, `KafkaProtobufCallback`, etc.)
  - File structure details
  - Import compatibility strategy
  - Extraction strategy (phased approach)
  - Code sharing mechanisms
  - Testing strategy

**Estimated Time**: 5-10 minutes (AI generation)

---

### Step 1.4: Review Generated Design

**Action**: Review the generated design document

```bash
# View generated design
cat .kiro/specs/kafka-proto-code-improvement/design.md
```

**Review Checklist**:
- [ ] Architecture aligns with requirements
- [ ] File structure matches improvement plan
- [ ] Class hierarchy is sound (KafkaBackendBase, etc.)
- [ ] Import compatibility strategy is comprehensive
- [ ] Extraction strategy is feasible (phased approach)
- [ ] Code sharing via base class is clear
- [ ] Testing strategy is defined

**Key Sections to Verify**:
1. **Module Structure**: Both Kafka and Protobuf modules clearly defined
2. **Class Design**: `KafkaBackendBase` for shared infrastructure
3. **Compatibility**: Root-level and module-level shims documented
4. **Extraction**: Phased approach (TopicManager → Partitioner → Headers)
5. **Testing**: Test migration strategy included

---

### Step 1.5: Approve Design

**Action**: Mark design as approved in `spec.json`

**Manual Update**:
```json
{
  "phases": {
    "design": {
      "status": "approved",  // Change from "pending"
      "approved_date": "2025-01-15"
    }
  }
}
```

**Or**: Use `-y` flag during task generation to auto-approve

**Status After Approval**: Ready for task generation

---

## Phase 2: Generate Tasks

### Step 2.1: Validate Design Approval

**Action**: Ensure design is approved before generating tasks

**Check**:
```bash
# Verify design status
cat .kiro/specs/kafka-proto-code-improvement/spec.json | grep -A 5 '"design"'
```

**Required**: `"status": "approved"` in design phase

---

### Step 2.2: Generate Tasks Document

**Command**:
```bash
kiro:spec-tasks kafka-proto-code-improvement
```

**Or with auto-approve**:
```bash
kiro:spec-tasks kafka-proto-code-improvement -y
```

**What This Does**:
1. Validates design exists and is approved
2. Reads context from:
   - `.kiro/specs/kafka-proto-code-improvement/requirements.md`
   - `.kiro/specs/kafka-proto-code-improvement/design.md`
   - `.kiro/specs/kafka-proto-code-improvement/spec.json`
   - `.kiro/steering/*.md` (project context)
   - `.kiro/settings/rules/tasks-generation.md` (task rules)
   - `.kiro/settings/templates/specs/tasks.md` (template)
3. Generates detailed task breakdown
4. Organizes tasks by phase
5. Estimates effort per task
6. Defines acceptance criteria
7. Updates `spec.json` with tasks phase status

**Expected Output**:
- Updated `tasks.md` with:
  - Phase 1.0: Legacy backend status update (0.5 days)
  - Phase 1.1: Protobuf reorganization (2-3 days, multiple tasks)
  - Phase 1.2: Kafka reorganization (4-5 days, phased extraction tasks)
  - Phase 1.3: Protobuf backend creation (3-4 days, multiple tasks)
  - Phase 1.4: Metrics & documentation (2-3 days)
  - Phase 1.5: Test migration (1-2 days)
  - Each task with:
    - Clear description
    - Acceptance criteria
    - Effort estimate
    - Dependencies
    - Requirement mapping (FR1, FR2, etc.)

**Estimated Time**: 5-10 minutes (AI generation)

---

### Step 2.3: Review Generated Tasks

**Action**: Review the generated tasks document

```bash
# View generated tasks
cat .kiro/specs/kafka-proto-code-improvement/tasks.md
```

**Review Checklist**:
- [ ] All requirements mapped to tasks
- [ ] Tasks properly sized (1-3 hours each, or 0.5-1 day for larger tasks)
- [ ] Task progression is logical
- [ ] Phases align with implementation plan
- [ ] Effort estimates are reasonable
- [ ] Acceptance criteria are clear
- [ ] Dependencies are identified
- [ ] Test tasks included

**Key Sections to Verify**:
1. **Phase Organization**: Tasks grouped by phase (1.0, 1.1, 1.2, etc.)
2. **Task Sizing**: Tasks are appropriately sized (not too large)
3. **Dependencies**: Clear task dependencies
4. **Acceptance Criteria**: Each task has measurable criteria
5. **Test Coverage**: Test tasks included for each phase

---

### Step 2.4: Approve Tasks

**Action**: Mark tasks as approved in `spec.json`

**Manual Update**:
```json
{
  "phases": {
    "tasks": {
      "status": "approved",  // Change from "pending"
      "approved_date": "2025-01-15",
      "total_tasks": <count>,
      "estimated_effort_days": 13
    }
  }
}
```

**Status After Approval**: Ready for implementation

---

## Phase 3: Implementation Readiness

### Step 3.1: Final Verification

**Checklist**:
- [ ] Requirements approved
- [ ] Design approved
- [ ] Tasks approved
- [ ] All documents reviewed
- [ ] Effort estimates validated
- [ ] Dependencies understood

---

### Step 3.2: Begin Implementation

**Command** (when ready to start):
```bash
kiro:spec-impl kafka-proto-code-improvement 1.0
```

**Note**: Start with Phase 1.0 (legacy backend status update) as it's the smallest task.

**Implementation Workflow**:
1. Execute tasks in order (1.0 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5)
2. Update task status as you complete them
3. Run tests after each phase
4. Update `spec.json` with progress

---

## Command Reference

### Design Generation
```bash
# Standard (requires approval)
kiro:spec-design kafka-proto-code-improvement

# Auto-approve requirements
kiro:spec-design kafka-proto-code-improvement -y
```

### Task Generation
```bash
# Standard (requires design approval)
kiro:spec-tasks kafka-proto-code-improvement

# Auto-approve design
kiro:spec-tasks kafka-proto-code-improvement -y
```

### Status Check
```bash
# Check spec status
kiro:spec-status kafka-proto-code-improvement
```

### Validation (Optional)
```bash
# Validate design quality
kiro:validate-design kafka-proto-code-improvement

# Validate implementation gap (if needed)
kiro:validate-gap kafka-proto-code-improvement
```

---

## Expected Timeline

| Phase | Command | Duration | Output |
|-------|---------|---------|--------|
| **Requirements Review** | Manual | 15-30 min | Approval |
| **Design Generation** | `kiro:spec-design` | 5-10 min | `design.md` |
| **Design Review** | Manual | 30-60 min | Approval |
| **Task Generation** | `kiro:spec-tasks` | 5-10 min | `tasks.md` |
| **Task Review** | Manual | 30-60 min | Approval |
| **Total** | - | **1.5-2.5 hours** | Ready for implementation |

---

## Success Criteria

After completing design and task generation:

1. ✅ `design.md` contains comprehensive architecture
2. ✅ `tasks.md` contains detailed task breakdown
3. ✅ All requirements mapped to tasks
4. ✅ Tasks properly sized and sequenced
5. ✅ Effort estimates validated
6. ✅ `spec.json` updated with phase statuses
7. ✅ Ready for implementation

---

## Troubleshooting

### Design Generation Issues

**Problem**: Design doesn't match requirements
**Solution**: 
- Review requirements for clarity
- Re-run `kiro:spec-design` (merge mode will use existing design)
- Provide feedback in requirements if needed

**Problem**: Missing architecture details
**Solution**:
- Check steering files for project context
- Verify design templates are available
- Re-run with more specific requirements

### Task Generation Issues

**Problem**: Tasks don't match design
**Solution**:
- Review design for completeness
- Re-run `kiro:spec-tasks` (merge mode)
- Verify design is approved in `spec.json`

**Problem**: Tasks too large or too small
**Solution**:
- Review task sizing rules in `.kiro/settings/rules/tasks-generation.md`
- Manually adjust task breakdown if needed
- Re-run task generation

---

## Next Steps After Generation

1. **Review Generated Documents**
   - Design: Architecture, class hierarchy, file structure
   - Tasks: Task breakdown, effort estimates, acceptance criteria

2. **Validate Against Improvement Plan**
   - Compare with `docs/KAFKA_PROTO_IMPROVEMENT_PLAN.md`
   - Ensure alignment with review findings
   - Verify protobuf reorganization included

3. **Approve and Proceed**
   - Mark phases as approved in `spec.json`
   - Begin implementation with Phase 1.0
   - Follow task order

---

## Quick Start Commands

```bash
# 1. Generate design (after requirements review)
kiro:spec-design kafka-proto-code-improvement

# 2. Review design, then generate tasks
kiro:spec-tasks kafka-proto-code-improvement

# 3. Check status anytime
kiro:spec-status kafka-proto-code-improvement

# 4. Begin implementation (after tasks approved)
kiro:spec-impl kafka-proto-code-improvement 1.0
```

---

**Status**: Ready for Design Generation  
**Next Action**: Run `kiro:spec-design kafka-proto-code-improvement`  
**Created**: January 15, 2025
