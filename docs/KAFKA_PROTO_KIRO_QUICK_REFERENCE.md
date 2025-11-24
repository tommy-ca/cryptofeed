# Kafka & Protobuf Code Improvement - Kiro Quick Reference

**One-page command reference**

---

## Current Status

✅ **Requirements**: Draft (ready)  
⏳ **Design**: Pending generation  
⏳ **Tasks**: Pending generation

---

## Generation Workflow

### 1️⃣ Generate Design
```bash
kiro:spec-design kafka-proto-code-improvement
```
**Prerequisites**: Requirements exist  
**Output**: `design.md` with architecture, class hierarchy, file structure  
**Time**: 5-10 minutes

### 2️⃣ Review Design
```bash
cat .kiro/specs/kafka-proto-code-improvement/design.md
```
**Check**: Architecture, file structure, class design, compatibility strategy

### 3️⃣ Approve Design
Update `spec.json` or use `-y` flag in next step

### 4️⃣ Generate Tasks
```bash
kiro:spec-tasks kafka-proto-code-improvement
```
**Prerequisites**: Design approved  
**Output**: `tasks.md` with detailed task breakdown by phase  
**Time**: 5-10 minutes

### 5️⃣ Review Tasks
```bash
cat .kiro/specs/kafka-proto-code-improvement/tasks.md
```
**Check**: Task sizing, dependencies, acceptance criteria, effort estimates

### 6️⃣ Approve Tasks
Update `spec.json` or proceed to implementation

---

## Auto-Approve Shortcuts

```bash
# Auto-approve requirements, generate design
kiro:spec-design kafka-proto-code-improvement -y

# Auto-approve design, generate tasks
kiro:spec-tasks kafka-proto-code-improvement -y
```

---

## Status Check

```bash
# Check spec status
kiro:spec-status kafka-proto-code-improvement

# View spec.json
cat .kiro/specs/kafka-proto-code-improvement/spec.json
```

---

## Optional Validation

```bash
# Validate design quality
kiro:validate-design kafka-proto-code-improvement

# Validate implementation gap (if needed)
kiro:validate-gap kafka-proto-code-improvement
```

---

## Expected Outputs

### Design Document (`design.md`)
- Architecture overview
- Module structure (Kafka + Protobuf)
- Class hierarchy (`KafkaBackendBase`, `KafkaProtobufCallback`)
- File organization
- Import compatibility strategy
- Extraction strategy (phased)
- Testing approach

### Tasks Document (`tasks.md`)
- Phase 1.0: Legacy backend status (0.5 days)
- Phase 1.1: Protobuf reorganization (2-3 days)
- Phase 1.2: Kafka reorganization (4-5 days)
- Phase 1.3: Protobuf backend (3-4 days)
- Phase 1.4: Metrics & docs (2-3 days)
- Phase 1.5: Test migration (1-2 days)
- Each task with: description, acceptance criteria, effort, dependencies

---

## Implementation Start

```bash
# After tasks approved, start implementation
kiro:spec-impl kafka-proto-code-improvement 1.0
```

---

**Full Plan**: See `docs/KAFKA_PROTO_KIRO_GENERATION_PLAN.md`  
**Workflow Guide**: See `docs/KAFKA_PROTO_KIRO_WORKFLOW.md`
