# Assignment 2 Submission Checklist

## ✅ Files to Submit

### Required (Minimum)
1. ✅ `assignment_2.ipynb` - Main demonstration notebook
2. ✅ `probability_tool.py` - Probability calculation module

### Supporting Files (if submitting complete package)
3. `../05_src/pokemon_chat/main.py` - Main module
4. `../05_src/pokemon_chat/prompts.py` - System prompts  
5. `../05_src/pokemon_chat/__init__.py` - Package init
6. `../05_src/utils/logger.py` - Logger utility

## 📋 Submission Steps

### Step 1: Verify Files
```bash
cd 02_activities
ls -la assignment_2.ipynb probability_tool.py
```

### Step 2: Test the Notebook
- Open `assignment_2.ipynb` in Jupyter
- Run all cells to ensure they execute
- Verify outputs are present

### Step 3: Clean Up (Optional)
- Remove output cells if required by submission guidelines
- Clear any sensitive API keys from notebook

### Step 4: Package Files
**Minimal submission:**
```bash
zip assignment_2_submission.zip \
  assignment_2.ipynb \
  probability_tool.py \
  README.md
```

**Complete submission:**
```bash
zip assignment_2_complete.zip \
  assignment_2.ipynb \
  probability_tool.py \
  ../05_src/pokemon_chat/*.py \
  ../05_src/utils/logger.py \
  README.md
```

### Step 5: Include Documentation
- ✅ README.md (created)
- ✅ Brief explanation of what each tool does
- ✅ Setup instructions for reviewer

## 📝 What the Assignment Demonstrates

1. **LangGraph Agent**: Multi-tool orchestration
2. **Three Tools**:
   - Card information (API integration)
   - Rules/deck building (vector search)
   - Probability calculator (mathematical calculations)
3. **Direct Probability Calculation**: No confirmation workflow
4. **Complete Integration**: All tools work together in chat interface

## ⚠️ Notes for Reviewer

- Notebook requires `../05_src/pokemon_chat/` module to run
- Needs API keys in `../05_src/.secrets` (not included)
- ChromaDB needed for rules/deck building tool
- All probability calculations are exact (no approximations)

