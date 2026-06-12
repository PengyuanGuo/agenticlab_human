## Coding Style and Principles

### Function-based Design
- **Prefer functions over classes**: Use standalone functions and functional programming patterns where possible
- Only use classes when managing stateful operations 
- Classes that act as callable function containers should implement `__call__()` to maintain functional interface

### Documentation
- Use concise single-line docstrings for simple functions
- Only add detailed docstrings when the function behavior is non-trivial or requires explanation
- Avoid redundant documentation that simply restates the function name

### Code Organization
- Break complex logic into focused, single-purpose functions
- Keep functions cohesive - extract logical units like transformation computations or data processing
- Use descriptive function names that clearly indicate purpose 
- Avoid over-fragmenting code into too many tiny functions

### Naming Conventions
- Use descriptive variable names that indicate transformations: 
- Follow the pattern `target_from_source` for transformation matrices
- Use `_fn` suffix for higher-order functions (e.g., `grasp_to_mat4x4_fn`)

### Type Hints
- Use type hints for function parameters and return values
- Leverage  for tensor dimensions 
- Define TypedDict classes for complex return structures 

### Error Handling
- Validate inputs at the start of functions with clear error messages
- Use `raise ValueError` or `raise RuntimeError` with descriptive messages
- Include context in error messages (e.g., which object/parameter caused the issue)

### Imports
- Group imports: standard library, third-party, local imports
- Use absolute imports from  package root
- Avoid wildcard imports
- Avoid local imports inside functions for standard modules; import at module level

### String Formatting
- Always use f-strings for string formatting; avoid `%s` or `.format()`


## Working Style

- **Discuss before implementing** — When the approach isn't obvious (e.g., where to put docs, how to structure config), talk through options before writing code
- **Comments and docstrings must be accurate** — Don't write comments that describe implementation details irrelevant to the reader or are vaguely wrong. If a comment doesn't add real information, drop it
- **Don't add dead code paths** — If every case goes down the same branch, don't add the other branch "just in case." Keep what's tested, remove what isn't

## Documentation Style Guide

When working on documentation:
- Be concise without making the writing feel unnatural
- Handle edge cases appropriately but not exhaustively
- Avoid excessive verbosity
- Use MyST-Parser Markdown features (colon fences, admonitions, etc.)
- Follow the existing black & white theme aesthetic
- API documentation will be auto-generated from docstrings once source code is added

## Key Concepts

