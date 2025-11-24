# Kafka & Protobuf Code Improvement - Kiro Workflow Guide

**How to use `kiro:spec-*` commands with this specification**

---

## Specification Created

✅ **Spec Name**: `kafka-proto-code-improvement`  
✅ **Location**: `.kiro/specs/kafka-proto-code-improvement/`  
✅ **Status**: Requirements Phase (ready for design generation)

---

## Kiro Command Workflow

### Step 1: Review Requirements

The requirements document is ready for review:
```bash
# View requirements
cat .kiro/specs/kafka-proto-code-improvement/requirements.md

# Or open in editor
code .kiro/specs/kafka-proto-code-improvement/requirements.md
```

**Current Status**: ✅ Requirements drafted and ready for approval

---

### Step 2: Generate Design

Once requirements are approved, generate the design:

```bash
kiro:spec-design kafka-proto-code-improvement
```

**What this does**:
- Analyzes requirements
- Generates detailed design document
- Creates architecture diagrams
- Defines class hierarchies
- Specifies file structures
- Documents import compatibility strategy

**Expected Output**: 
- Updated `design.md` with comprehensive design
- Architecture diagrams
- Class designs
- Module structure details

---

### Step 3: Review and Approve Design

Review the generated design:
```bash
# View design
cat .kiro/specs/kafka-proto-code-improvement/design.md
```

**Review Checklist**:
- [ ] Architecture aligns with requirements
- [ ] File structure is clear
- [ ] Class hierarchy makes sense
- [ ] Import compatibility strategy is sound
- [ ] Extraction strategy is feasible

**Approve Design**: Once satisfied, mark design as approved in `spec.json`

---

### Step 4: Generate Tasks

After design approval, generate detailed tasks:

```bash
kiro:spec-tasks kafka-proto-code-improvement
```

**What this does**:
- Breaks down design into actionable tasks
- Organizes tasks by phase
- Estimates effort for each task
- Defines acceptance criteria
- Creates task dependencies

**Expected Output**:
- Updated `tasks.md` with detailed task breakdown
- Tasks organized by phase (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
- Effort estimates
- Acceptance criteria per task

---

### Step 5: Execute Tasks

Work through tasks in order:

```bash
# View tasks
cat .kiro/specs/kafka-proto-code-improvement/tasks.md

# Mark tasks as complete as you work through them
# Update spec.json status as phases complete
```

---

## Specification Structure

```
.kiro/specs/kafka-proto-code-improvement/
├── spec.json              # Specification metadata
├── requirements.md        # ✅ Draft (ready)
├── design.md              # ⏳ Pending (generate with kiro:spec-design)
├── tasks.md               # ⏳ Pending (generate with kiro:spec-tasks)
└── README.md              # Quick reference
```

---

## Current Status

| Phase | Status | Command | Next Action |
|-------|--------|---------|-------------|
| **Requirements** | ✅ Draft | - | Review and approve |
| **Design** | ⏳ Pending | `kiro:spec-design` | Generate design |
| **Tasks** | ⏳ Pending | `kiro:spec-tasks` | Generate after design approval |

---

## Quick Reference Commands

```bash
# Generate design (after requirements approval)
kiro:spec-design kafka-proto-code-improvement

# Generate tasks (after design approval)
kiro:spec-tasks kafka-proto-code-improvement

# View spec status
cat .kiro/specs/kafka-proto-code-improvement/spec.json

# View requirements
cat .kiro/specs/kafka-proto-code-improvement/requirements.md

# View design (after generation)
cat .kiro/specs/kafka-proto-code-improvement/design.md

# View tasks (after generation)
cat .kiro/specs/kafka-proto-code-improvement/tasks.md
```

---

## Workflow Summary

```
1. Review Requirements ✅
   └─> Approve requirements

2. Generate Design
   └─> kiro:spec-design kafka-proto-code-improvement
   └─> Review generated design
   └─> Approve design

3. Generate Tasks
   └─> kiro:spec-tasks kafka-proto-code-improvement
   └─> Review generated tasks
   └─> Begin implementation

4. Execute Tasks
   └─> Work through tasks in order
   └─> Update status as phases complete
```

---

## Related Documentation

- **Full Improvement Plan**: `docs/KAFKA_PROTO_IMPROVEMENT_PLAN.md`
- **Review Findings**: `docs/KAFKA_PROTO_IMPROVEMENT_PLAN_REVIEW.md`
- **Quick Summary**: `docs/KAFKA_PROTO_IMPROVEMENT_SUMMARY.md`
- **Migration Checklist**: `docs/KAFKA_PROTO_MIGRATION_CHECKLIST.md`

---

**Ready to Proceed**: Review requirements, then run `kiro:spec-design kafka-proto-code-improvement`
