# Hybrid Code Clone Detection Using Multi-Technique Analysis

A hybrid code clone detection system that combines **text-based, token-based, structural, and semantic analysis** to identify similar or duplicated code fragments. The project explores a multi-technique approach to detecting code clones, including semantically similar code that may differ in syntax or structure.

## 📌 Overview

Code cloning occurs when identical or similar code fragments appear in multiple locations within a software project. Although code reuse can be useful, unmanaged code clones can increase maintenance costs, introduce bugs, and make software evolution more difficult.

This project explores a hybrid detection approach that combines multiple similarity signals to identify different types of code clones. It incorporates **CodeBERT-based semantic representations** alongside traditional code comparison techniques.

## 🎯 Objectives

- Detect code clones using multiple complementary techniques.
- Identify exact, syntactically modified, and semantically similar code fragments.
- Compare code using textual, token-based, structural, and semantic features.
- Combine similarity scores into a unified detection pipeline.
- Evaluate detection performance using standard classification metrics.

## ⚙️ Detection Techniques

The proposed approach combines four techniques with the following weights, as described in the research paper.

| Technique | Weight | Purpose |
|---|---:|---|
| Text-based analysis | 0.20 | Compares source-code text and sequences. |
| Token-based analysis | 0.25 | Compares tokens and normalized identifiers. |
| Structural analysis | 0.20 | Examines similarities in code structure and control-flow patterns. |
| Semantic analysis | 0.35 | Uses CodeBERT-based representations to compare semantic similarity. |

### 1. Text-Based Analysis

Compares source-code fragments to identify identical or nearly identical code. It provides a baseline for detecting straightforward code duplication.

### 2. Token-Based Analysis

Converts source code into tokens and can normalize identifiers to reduce the effect of variable renaming and other superficial changes.

### 3. Structural Analysis

Examines structural patterns in code, such as loops, conditional statements, and other programming constructs, to identify similarities that may not be obvious from raw text.

### 4. Semantic Analysis Using CodeBERT

Uses CodeBERT-based representations to estimate semantic similarity between code fragments. This approach aims to identify code that performs similar operations despite differences in syntax or implementation.

## 🧠 Hybrid Similarity Model

The proposed model combines the four similarity signals using their respective weights.

The weighted similarity score can be expressed as:

\[
S = 0.20S_t + 0.25S_k + 0.20S_s + 0.35S_m
\]

Where:

- \(S_t\): Text-based similarity
- \(S_k\): Token-based similarity
- \(S_s\): Structural similarity
- \(S_m\): Semantic similarity

The resulting score can be compared against a selected threshold to determine whether two code fragments are sufficiently similar to be flagged as potential clones.

*Note: This equation describes the weighted approach implied by the paper's technique weights. The exact normalization, threshold selection, and implementation should match the actual project code.*

## 🔍 Types of Code Clones

The project targets the following commonly recognized clone categories:

| Clone type | Description |
|---|---|
| Type-1 | Identical code, apart from formatting and comments. |
| Type-2 | Structurally similar code with changes such as identifier or literal renaming. |
| Type-3 | Code with added, removed, or modified statements. |
| Type-4 | Code that implements similar functionality but differs syntactically. |

The effectiveness of detection for each category depends on the implemented comparison techniques and evaluation dataset.

## 📊 Evaluation Metrics

The research paper reports the following performance metrics in its evaluation table:

| Metric | Reported value |
|---|---:|
| Accuracy | 73% |
| Precision | 0.72 |
| Recall | 0.69 |
| F1 Score | 0.70 |

These figures are reported results from the paper, not independently verified benchmark results. The paper also contains a separate approximate accuracy estimate in its comparison table, so the evaluation context should be considered when interpreting the numbers.

## 🛠️ Technology Stack

- **Programming language:** Python
- **Semantic representation:** CodeBERT
- **Machine learning / NLP:** Transformer-based code representations
- **Core techniques:** Text similarity, token analysis, structural comparison, semantic similarity
- **Evaluation:** Accuracy, precision, recall, and F1 score

The exact libraries and versions should be documented according to the project's current dependency files.

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- The project's required Python packages
- Access to the model weights or model repository required by the CodeBERT implementation

### 1. Clone the repository

```bash
git clone https://github.com/suyashvishnoi13/code-clone-detector.git
cd code-clone-detector
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If the repository includes a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

If it does not, install the dependencies specified by the actual implementation before running the project.

### 4. Run the application

Use the entry point documented in the repository. For example, if the project uses a Python script as its entry point:



## 💡 Potential Applications

- Identifying duplicated code during software maintenance.
- Supporting code review and refactoring.
- Studying similarities between code fragments.
- Detecting potentially redundant implementations.
- Supporting software quality analysis and code reuse investigations.

These are potential applications of the approach rather than claims that every feature is implemented in the current repository.

## 🔮 Future Improvements

- Support additional programming languages.
- Improve structural comparison using abstract syntax trees.
- Tune similarity weights using validation data.
- Optimize semantic comparison for large repositories.
- Add interactive visualization of detected clone pairs.
- Evaluate on established code-clone benchmarks.
- Improve scalability for large codebases.

## 👨‍💻 Author

**Suyash Vishnoi**

- GitHub: [@suyashvishnoi13](https://github.com/suyashvishnoi13)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/suyash-vishnoi-a98179343/)

## 📄 License

No license is specified here. Add a `LICENSE` file and update this section if you intend to distribute the project under a particular open-source license.
