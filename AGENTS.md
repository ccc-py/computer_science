# AGENTS.md

## Structure

This is a personal educational repository with three main directories:
- `theory/` - Computability theory implementations (Turing machines, lambda calculus, finite state machines, grammars)
- `interpreter/` - Language interpreter implementations (BASIC, JavaScript, Lisp, Prolog, Python)
- `ai/` - Various AI/agent experiments (neural networks, multi-agent systems)

## Running Tests

Each subdirectory has its own `test.sh`. There is no centralized test runner.

```bash
# Run specific test
./interpreter/py0i/test.sh
./interpreter/basic/test.sh
./theory/lambda/03-interpreter/test.sh
```

## Running Interpreters

```bash
python3 interpreter/basic/basic.py interpreter/basic/bas/hello.bas
node interpreter/js0i/js0i.js script.js
python3 interpreter/lisp/lisp.py program.lisp
python3 interpreter/prolog/prolog.py program.pl
python3 interpreter/py0i/py0i.py script.py
```

## Notes

- No package managers or build systems - plain Python/JavaScript scripts
- Written primarily in Chinese (Traditional)
- Many subdirectories contain experimental/bak versions - check before editing