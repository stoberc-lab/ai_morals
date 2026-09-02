library(tidyverse)

#===============================================================================
# Requirement 
#
# variables = block, left, right, prob_1, prob_2
# ==============================================================================

# ==============================================================================
# prepare_wst_data()

# Taking raw csv fdata and reshpae to wide format where: 
#   - each row is one block 
#   - each column is one pair(e.g., "A_CE")
#   - value in each cell is p_val = prob_1 / (prob_1 + prob_2) 
# ==============================================================================

prepare_wst_data <- function(data) {
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
# test_wst()
# 
# Takes wide-format (prepare_wst_data()) and tests the WST condition for 
# every possible triad
#
# Checking violation: 
# P(A > B) >= 0.5 AND P(B > C) >= 0.5 BUT P(A > C) < 0.5 (reverse cycle)
#
# get_p(): given any two item x and y, find correct column regardless of which 
# direction it was stored in file and flips probability if needed so it 
# always returns P(x > y)              [NEEDED FOR COUNTERBALANCE]
# ==============================================================================

test_wst <- function(df_wide) {
  # Generate every possible 3-item combination (35 triads)
  items <- attr(df_wide, "items")
  triads <- t(combn(items, 3))
  
  checked_df <- df_wide
  wst_cols <- c() 
  
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

# Test WST for each traid 
  for (i in 1:nrow(triads)) {
    a <- triads[i, 1]
    b <- triads[i, 2]
    c <- triads[i, 3]
  
    triad_name <- paste("wst",a,b,c,sep="_")
    wst_cols <- c(wst_cols, triad_name)
    
    # get_p (always getting P(X > Y) in original direction)
    p_ab <- get_p(checked_df, a, b)
    p_bc <- get_p(checked_df, b, c)
    p_ac <- get_p(checked_df, a, c)
    
    # WST Violation
    intransitive_condition <- 
      (p_ab >= 0.5 & p_bc >= 0.5 & p_ac < 0.5) |
      ((1-p_ab) >= 0.5 & (1-p_bc) >= 0.5 & (1-p_ac) < 0.5)
    
    checked_df[[triad_name]] <- as.integer(intransitive_condition)
  }

  # WST_check_total: 1 = no violation across all triad 
  # WST_check_total: 0 = at least one violation
  checked_df$WST_check_total <- ifelse(
    rowSums(checked_df[, wst_cols]) == 0,1,0
)

return(checked_df)
}

# ==============================================================================
# add_preference_rank()
# 
# Calculate preference ranks from raw data and merges it to wide WST data
#===============================================================================
add_preference_rank <- function(raw_df, wst_df) {
  # Calculate wins from raw data
  wins_df <- raw_df %>%
    mutate(
      p_val = prob_1 / (prob_1 + prob_2),
      left_win = ifelse(p_val >= 0.5, 1, 0),
      right_win = ifelse(p_val <= 0.5, 1, 0)
    ) %>%
    select(block, left, right, left_win, right_win) %>%
    pivot_longer(
      cols = c(left, right), 
      names_to = "side", 
      values_to = "item"
    ) %>%
    mutate(win_point = ifelse(side == "left", left_win, right_win)) %>%
    group_by(block, item) %>%
    summarise(total_wins = sum(win_point), .groups = "drop")
  
  # Generate the rank string per block
  ranks_per_block <- wins_df %>%
    group_by(block) %>%
    arrange(desc(total_wins), item, .by_group = TRUE) %>%
    summarise(
      preference_rank = paste(item, collapse = " > "),
      .groups = "drop"
    )
  
  # 3. Merge rankings into the WST results
  final_df <- wst_df %>%
    left_join(ranks_per_block, by = "block") %>%
    # Mask the rank with NA if the block has WST violation
    mutate(preference_rank = ifelse(WST_check_total == 1, preference_rank, NA))
  
  return(final_df)
}


# ==============================================================================
# run_wst() 
#
# wrapper that tie everything together: 
#   1. read csv 
#   2. prepare wide format 
#   3. add rankings 
#   4. run WST 
#   5. save results and print summary 
# ==============================================================================

run_wst <- function(input, result) {
  data    <- read_csv(input, show_col_types = FALSE)
  df_wide <- prepare_wst_data(data)
  wst_out <- test_wst(df_wide)
  
  # apply newly added ranking function 
  wst_out <- add_preference_rank(raw_df = data, wst_df = wst_out)
  
  # grab block, WST_check_total, then all wst triad columns
  wst_cols <- names(wst_out)[startsWith(names(wst_out), "wst_")]
  
  results <- wst_out %>%
    select(block, WST_check_total, preference_rank, all_of(wst_cols))
  
  write.csv(results, result, row.names = FALSE)
  invisible(results)
}