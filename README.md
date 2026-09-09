# Repository Structure
```
|-- input/
     -- OriginalItemSet_edited.csv      # Input data file w/ 126 moral vignette tradeoffs
|-- llm_elicitation/                   
     -- *.py                            # Scripts for running the LLM elicitation (*: deepseek, gemini, llama4, openai, xai)
|-- output/                            
     -- *.csv                           # Output files with the LLM elicitation results (*: deepseek, gemini, llama4, openai, xai)
|-- transitivity_result/                 
     -- WUM_Adherence.xlsx              # Weak Utility Model (WUM)
     -- HCM_Adherence.xlsx              # High Certainty Model (HCM)
     -- G-RUM_Adherence.xlsx            # General Random Utility Model (G-RUM)
|-- transitivity_test/                 
     -- WUM_analysis_function.R         # Weak Utility Model (WUM)
     -- HCM_analysis_function.R         # High Certainty Model (HCM)
     -- G-RUM_analysis_function.R       # General Random Utility Model (G-RUM)
```

# Replication Steps
## Step 1: Run the Python scripts for each of the LLMs
Use the Python scripts in the "llm_elicitation" folder. The scripts will need to input "OriginalItemSet_edited.csv" located in the "input" folder.

## Step 2: Review the outputs
The outputs for each of the LLMs can be found under the "output" folder.  

## Step 3: Test the transitivity models against the outputs  
### Convert the output files into the correct data format 
- `block` : Groups the comparisons that are tested together. One row of output per block (e.g., 1,2,...6)
- `left` : The moral principle alternative presented first (e.g., cp, f, ... a)
- `right` : The moral principle alternative presented second (e.g., a, cp, ... f)
- `prob_1` : Probability assigned to the `left` alternative
- `prob_2` : Probability assigned to the `right` alternative

Probabilities are normalized within each row as `prob_1 / (prob_1 + prob_2)`, so raw values can be passed in as they come out of the LLMs. 
The normalized value is read as P(`left` chosen over `right`) - in our case, the probability that the model judged the left situation to be worse violation. 

A block should contain all 21 pairs of the seven alternatives. Pairs can be stored in either direction. `get_p()` finds the pair in the order it was stored and flips the probability when it finds the reverse, so counterbalanced pairs need no reshaping before they are read in. Note that each file should contain a single presentation order. To analyze the original and counterbalanced orders separately, split the CSV file first. 

### Testing
Use the R scripts in "transitivity_test". The scripts will read in the LLM responses and test them against the transitivity models.  
These files test whether a set of pairwise choice proabilities is consistent with a transitive ordering. 
The WUM and HCM test whether one fixed ordering can account for the choices. G-RUM tests whether the choices could have come from sampling across several orderings. 
Each script reads a CSV file with five columns. 

`WUM_analysis_function.R`
- `prepare_wum_data(data)` normalizes each row and reshapes the file so that one row is one block and one column is one pair (named `left_right`).
- `test_wum(df_wide)` builds all 35 triads of the seven alternatives and check each one against the threshold. A triad is flagged when P(A>B) >= 0.5 and P(B > C) >= 0.5 but P(A > C) < 0.5 or when the same patter appears in the reversed direction. Each triad gets 0/1 column, and `WUM_check_total` is 1 only when no triad was flagged.
- `add_preference_rank(raw_df, wum_df)` derives the ordering. Each alternative earns a point for every pairwise comparison it wins. Points are summed within a block, and alternatives are ordered from most to least wins. Blocks that violate the test are left with a missing rank, since no single ordering describes them.

`HCM_analysis_function.R`
- `prepare_hcm_data(data)` runs exactly as WUM.
- `test_hcm(df_wide)` builds the same 35 triads and evaluates against the 0.9 threshold. However, unlike WUM this asks whetehr the three probabilities in a triad are consistent with one of the six possible orderings at 0.9 or above. All six patterns are check, unlike the WUM. `HCM_check_total` is 1 when all 35 triads are satisfied and 0 when any is not.

`G-RUM_analysis_function.R`

### Output
The output files with the transitivity test results can be found in the "transitivity_result" folder.
