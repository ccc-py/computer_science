# 電腦科學 (Computer Science)

本專案收錄計算理論與語言直譯器的經典實作，是學習電腦科學核心概念的個人教材。

## 目錄

| 目錄 | 說明 |
|------|------|
| [theory/](theory/README.md) | 計算理論實現 |
| [interpreter/](interpreter/README.md) | 語言直譯器集合 |
| [ai/](ai/README.md) | AI/Agent 實驗 |

---

## 計算理論 (Theory)

從有限狀態機到圖靈機，展示計算的本質與極限。

| 主題 | 說明 |
|------|------|
| [finiteStateMachine/](theory/finiteStateMachine/README.md) | 有限狀態機 (DFA) |
| [grammar/](theory/grammar/README.md) | 形式文法 (Chomsky 階層) |
| [lambda/](theory/lambda/README.md) | Lambda 演算 |
| [turingMachine/](theory/turingMachine/README.md) | 圖靈機 |

```
┌─────────────────────────────────────┐
│         Turing Machine               │
├─────────────────────────────────────┤
│         Lambda Calculus             │
├─────────────────────────────────────┤
│      Context-Free Grammar          │
├─────────────────────────────────────┤
│         Regular FSM                 │
└─────────────────────────────────────┘
```

### 核心問題

1. **停機問題** - 不存在程式能判斷任意程式是否停止
2. **通用圖靈機** - 可模擬任何其他圖靈機
3. **Church-Turing 論點** - Lambda 與圖靈機計算能力相同

---

## 直譯器 (Interpreters)

五種語言的直譯器實作，展示不同典範：

| 直譯器 | 語言 | 行數 | 特色 |
|--------|------|------|------|
| [basic/](interpreter/basic/README.md) | BASIC | 129 | 行號導向、簡單直譯 |
| [js0i/](interpreter/js0i/README.md) | JavaScript | 635 | 完整語法樹、Tree-walking |
| [lisp/](interpreter/lisp/README.md) | Lisp | 101 | S-expression、元程式 |
| [prolog/](interpreter/prolog/README.md) | Prolog | 280 | 模式比對、Backtracking |
| [py0i/](interpreter/py0i/README.md) | Python | 1017 | AST 直譯、完整標準庫 |

### 直譯器架構

```
原始碼 → Lexer → Parser → AST → Evaluator → 輸出
```

---

## AI 實驗

| 目錄 | 說明 |
|------|------|
| [nn0/](ai/README.md) | 神經網路實現 |
| [agent0/](ai/README.md) | Multi-Agent 系統 |

---

## 執行方式

```bash
# 直譯器
python3 interpreter/basic/basic.py interpreter/basic/bas/hello.bas
node interpreter/js0i/js0i.js script.js
python3 interpreter/lisp/lisp.py program.lisp
python3 interpreter/prolog/prolog.py program.pl
python3 interpreter/py0i/py0i.py script.py

# 測試
./interpreter/py0i/test.sh
./interpreter/basic/test.sh
./theory/lambda/03-interpreter/test.sh
```

---

## 為什麼學習這些？

- **語言設計**: 正規文法 → 語法分析 → 編譯原理
- **計算極限**: 什麼能算、什麼不能算
- **抽象思維**: 從具體機器到抽象模型
- **實現能力**: 從理論到代碼的轉換