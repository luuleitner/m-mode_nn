# Claude Instructions for This Project

> **Read this file first** before generating any code in this repository.

## Project Context

- **Languages**: Python, Embedded C, C++, Verilog
- **Domain**: Embedded systems, DSP, FPGA
- **Current Phase**: [RESEARCH / PRODUCTION / MIXED]

## Mandatory Behavior

### 1. Phase Detection (ALWAYS FIRST)

Before any code generation, identify the project phase:

**Research/Prototype Mode**
- Triggered by: prototype, experiment, POC, exploratory, sandbox
- Standards: Fast iteration, minimal ceremony, type hints ONLY on public APIs
- Trade-offs: Skip comprehensive error handling, focus on algorithm/logic
- Output: Scripts, notebooks, inline comments

**Production Mode**
- Triggered by: production, deploy, release, enterprise, mission-critical
- Standards: ≥80% test coverage, full error handling, comprehensive documentation
- Requirements: Type safety everywhere, static analysis clean, memory-safe
- Output: Proper modules, formal docs, CI/CD ready

**If phase is ambiguous → ASK EXPLICITLY**

---

### 2. Implementation Planning (MANDATORY)

For ANY code change beyond single-line edits, provide structured plan FIRST:

#### Required Plan Sections:

1. **Objective**
   - Single sentence goal
   - Measurable success criteria

2. **Current State Analysis**
   - Affected files with actual paths
   - Existing implementation (if modifying)
   - Dependencies and constraints

3. **State-of-the-Art Review** ⭐ CRITICAL
   
   For the specific problem domain:
   
   - **Industry standard approaches**
     - How is this solved in production? (Linux kernel, vendor SDKs, popular libraries)
     - Reference actual implementations
   
   - **Algorithm/architecture comparison**
     - List established methods
     - Academic/research papers if relevant
     - Performance characteristics of each
   
   - **Quantitative trade-off analysis**
     - Time complexity vs space complexity (Big-O)
     - Latency vs throughput (measured in μs, MB/s)
     - Accuracy vs computational cost
     - Simplicity vs flexibility
   
   - **Justification for our case**
     - Which approach fits our constraints?
     - Why chosen over alternatives?

4. **Proposed Implementation**
   - Chosen approach with explicit justification
   - How it compares to alternatives (QUANTITATIVELY)
   - Where it deviates from standard approaches and WHY

5. **Implementation Steps**
   - Numbered sequence
   - File path and operation for each step
   - Exact changes (function signatures, algorithms)
   - Purpose of each step

6. **Verification**
   - Tests with expected results
   - Performance benchmarks with baseline comparison
   - How to validate correctness

7. **Critical Risks**
   - Concrete failure modes (not hypotheticals)
   - Breaking changes
   - Rollback strategy

#### Quantitative Requirements:

**MUST include:**
- Specific algorithm names (e.g., "Bresenham's algorithm", NOT "line drawing")
- Measured or calculated performance (latency in μs, throughput in MB/s, area in LUTs)
- Resource usage (memory bytes, CPU cycles, FPGA resources)
- References to standards, papers, or production implementations

**MUST avoid:**
- Vague terms: "faster", "better", "optimized" without numbers
- Unverified claims about performance
- Assumptions without stating them explicitly
- Proposing approaches without checking feasibility

#### Critical Analysis Checklist:

Before presenting plan, answer:
1. ✓ What's the standard solution? (cite: Linux, vendor HAL, library, paper)
2. ✓ Why not use that? (if deviating, explicit reason)
3. ✓ What's the performance? (measured or calculated from first principles)
4. ✓ What's the complexity? (Big-O, resource usage, maintenance burden)
5. ✓ What could go wrong? (concrete failure modes)

**After plan approval → Execute completely without additional prompts**

---

### 3. Language-Specific Standards

#### Python
**Research Mode:**
- NO type hints except public API entry points
- stdlib preferred over third-party when possible
- Notebooks acceptable
- Inline comments, not docstrings

**Production Mode:**
- Type hints on ALL public functions
- Context managers for resources
- Proper exception hierarchy
- Logging (not print) for diagnostics
- Reference PEPs for decisions

**Always:**
- State GIL implications for concurrency
- Compare stdlib vs third-party for common tasks
- Measure performance implications of language features

#### Embedded C
**Research Mode:**
- Focus on algorithm correctness
- Compiler warnings OK
- Basic assertions for validation

**Production Mode:**
- Memory calculations: stack, heap, static (in bytes)
- Timing analysis: instruction counts or cycle measurements
- No memory leaks (valgrind clean if applicable)
- Proper resource cleanup

**Always:**
- Reference vendor documentation and datasheets
- Check hardware resource conflicts (DMA channels, timers, peripherals)
- Calculate worst-case stack usage
- Consider interrupt latency impact

#### C++
**Research Mode:**
- Readability over optimization
- Modern C++ features for clarity

**Production Mode:**
- const correctness everywhere
- RAII for all resource management
- Smart pointers for dynamic allocation
- No memory leaks (valgrind clean)
- No undefined behavior (sanitizers clean)

**Always:**
- Specify C++ standard version (C++11/14/17/20)
- Measure abstraction overhead (templates, virtual calls)
- Reference established libraries (Boost, STL, abseil)
- Consider compile-time vs runtime trade-offs

#### Verilog
**Research Mode:**
- Basic testbenches
- Waveform inspection over formal coverage
- Focus on logic correctness

**Production Mode:**
- Testbench covers all state transitions
- Timing constraints met
- Synthesis clean (no unintended latches)
- Formal verification for critical blocks

**Always:**
- Predict resource utilization (LUTs, FFs, BRAMs, DSPs)
- Timing analysis and critical path identification
- Reference IEEE standards and vendor app notes
- Check clock domain crossing (CDC) safety
- Reference: Cliff Cummings papers for CDC/FIFO design

---

### 4. General Principles

- **Be logical**: Follow sound engineering reasoning
- **Be clear**: Organize and structure all answers systematically
- **Be critical**: Challenge assumptions, point out potential issues
- **Be precise**: Use actual data, not estimates or generic claims
- **Compare alternatives**: Never propose a solution in isolation
- **Quantify everything**: "Faster" means nothing without measurements
- **Categorize systematically**: Group related items, sort by priority/importance

---

## Example Interaction

❌ **BAD:**
```
User: "Write a CSV parser"
Claude: [Generates code immediately]
```

✅ **GOOD:**
```
User: "Write a CSV parser"
Claude: "Is this for research/experimentation or production use? This affects 
         error handling, testing requirements, and implementation approach."
User: "Production"
Claude: [Provides implementation plan with SoA review comparing csv module, 
         pandas, manual parsing, with quantitative trade-offs, then waits 
         for approval]
```

---

## Notes

- This file should be read at the START of every Claude session
- Update `Current Phase` section as project evolves
- Add project-specific constraints or requirements below
- Commit this file to version control for team consistency

---

## Project-Specific Additions

[Add any project-specific requirements here, e.g.:]
- Hardware: STM32F4 @ 168MHz, 128KB RAM
- Key constraints: Real-time audio, <1ms latency
- Existing codebase: HAL drivers, RTOS (FreeRTOS)
- Critical paths: Audio callback, DSP processing