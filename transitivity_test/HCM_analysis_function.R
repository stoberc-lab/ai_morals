library(tidyverse)

#===============================================================================
# Requirement 
#
# variables = block, left, right, prob_1, prob_2
# ==============================================================================

# ==============================================================================
# prepare_hct_data()

# Taking raw csv fdata and reshpae to wide format where: 
#   - each row is one block 
#   - each column is one pair(e.g., "A_CE")
#   - value in each cell is p_val = prob_1 / (prob_1 + prob_2) 
# ==============================================================================

prepare_hct_data <- function(data) {
  names(data) <- trimws(names(data))
  
  df <- data %>% 
    mutate(
      # normalize probability 
      p_val = prob_1 / (prob_1 + prob_2),
      pair_name = paste(left, right, sep = "_")
    )
  
  # Extract unique items (left and right columns)
  items <- unique(c(df$left, df$right))
  
  # Reshape to wide (one row per block)
  df_wide <- df %>% 
    select(block, pair_name, p_val) %>% 
    pivot_wider(names_from = pair_name, values_from = p_val)
  
  attr(df_wide, "items") <- items
  
  return(df_wide)
}


# ==============================================================================
# test_hct()
# 
# Takes wide-format (prepare_hct_data()) and tests the hct condition for 
# every possible triad
#
# Checking violation: 
# P(A > B) >= 0.5 AND P(B > C) >= 0.5 BUT P(A > C) < 0.5 (reverse cycle)
#
# get_p(): given any two item x and y, find correct column regardless of which 
# direction it was stored in file and flips probability if needed so it 
# always returns P(x > y)              [NEEDED FOR COUNTERBALANCE]
# ==============================================================================

test_hct <- function(df_wide) {
  # Generate every possible 3-item combination (35 triads)
  items <- attr(df_wide, "items")
  triads <- t(combn(items, 3))
  
  checked_df <- df_wide
  hct_cols <- c() 
  
  # get_p() 
  get_p <- function(data, x, y) {
    col_xy <- paste(x, y, sep = "_") # original
    col_yx <- paste(y, x, sep = "_") # counterbalanced
    
    if (col_xy %in% names(data)) {
      return(data[[col_xy]])             # original -> just return 
    } else if (col_yx %in% names(data)) { 
      return(1 - data[[col_yx]])         # counterbalance -> flip  
    } else {
      stop(paste("Can't find pair:", x, "vs", y))
    }
  }
  
  # Test hct for each traid 
  for (i in 1:nrow(triads)) {
    a <- triads[i, 1]
    b <- triads[i, 2]
    c <- triads[i, 3]
    
    triad_name <- paste("hct",a,b,c,sep="_")
    hct_cols <- c(hct_cols, triad_name)
    
    # get_p (always getting P(X > Y) in original direction)
    p_ab <- get_p(checked_df, a, b)
    p_bc <- get_p(checked_df, b, c)
    p_ac <- get_p(checked_df, a, c)
    
    # HCT Violation
    transitive_condition <-
      (p_ab >= 0.9 & p_bc >= 0.9 & p_ac >= 0.9)                       | # A > B > C
      (p_ac >= 0.9 & (1 - p_bc) >= 0.9 & p_ab >= 0.9)                 | # A > C > B
      ((1 - p_ab) >= 0.9 & p_ac >= 0.9 & p_bc >= 0.9)                 | # B > A > C
      (p_bc >= 0.9 & (1 - p_ac) >= 0.9 & (1 - p_ab) >= 0.9)           | # B > C > A
      ((1 - p_ac) >= 0.9 & p_ab >= 0.9 & (1 - p_bc) >= 0.9)           | # C > A > B
      ((1 - p_bc) >= 0.9 & (1 - p_ab) >= 0.9 & (1 - p_ac) >= 0.9)      # C > B > A
    
    checked_df[[triad_name]] <- as.integer(transitive_condition)
  }
  
  # HCT_check_total: 1 = all triads transitive, 0 = at least one is not
  # note: sum must equal total number of triads (not 0 like WST)
  n_triads <- nrow(triads)
  checked_df$HCT_check_total <- ifelse(
    rowSums(checked_df[, hct_cols]) == n_triads, 1, 0
  )
  
  return(checked_df)
}

# ==============================================================================
# run_hct() 
#
# wrapper that tie everything together: 
#   1. read csv 
#   2. prepare wide format 
#   3. run WST 
#   4. save results and print summary 
# ==============================================================================

run_hct <- function(input, result) {
  data    <- read_csv(input, show_col_types = FALSE)
  df_wide <- prepare_hct_data(data)
  hct_out <- test_hct(df_wide)
  
  # grab block, WST_check_total, then all wst triad columns
  hct_cols <- names(hct_out)[startsWith(names(hct_out), "hct_")]
  
  results <- hct_out %>%
    select(block, HCT_check_total, all_of(hct_cols))
  
  write.csv(results, result, row.names = FALSE)
  
  print(results %>% select(block, HCT_check_total))
  invisible(results)
}
