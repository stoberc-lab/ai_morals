library(tidyverse)
library(combinat)
library(lpSolve)

# ==============================================================================
# 1. Generate Transitive Vertices
# ==============================================================================

generate_polytope_vertices <- function() {
  # generate all linear order (7! = 5040)
  # permn = return list
  # do.call(rbind, ...) turn it into a 5040 x 7 matrix
  C <- 1:7
  
  P <- do.call(rbind, permn(C))
  
  # Pre-allocate V matrix
  V <- matrix(0, nrow = 5040, ncol = 21)
  
  # Generate all 21 possible pairs
  pairs <- combn(C, 2)
  
  # Vectorize pair checking 
  for (i in 1:21) {
    a <- pairs[1, i]
    b <- pairs[2, i]
    
    # max.col efficiently find exact column index of 'a' and 'b' across all 5040 rows
    pos_a <- max.col(P == a)
    pos_b <- max.col(P == b)
    
    # if (pos_a > pos_b) give 1 or else 0 
    V[, i] <- ifelse(pos_a > pos_b, 1, 0)
  }
  
  return(V)
}

# ==============================================================================
# 2. Check if a single Sample (21 probabilities) is INSIDE polytope
# ==============================================================================

check_sample_in_polytope <- function(sample_probs, V) {
  # Objective function for linprog: minimize sum(x)
  f.obj <- rep(1, 5040)
  
  # Equality constraints: V' * x = Sample, and sum(x) = 1
  f.con <- rbind(t(V), rep(1, 5040))
  f.dir <- rep("==", 22)
  f.rhs <- c(sample_probs, 1)
  
  #lpSolve solves the linear program
  # NOTE: R automatically enforces x>=0.
  ## Because sum(x)==1, x <= 1 is guaranteed
  res <- lp(direction = "min",
            objective.in = f.obj,
            const.mat = f.con,
            const.dir = f.dir,
            const.rhs = f.rhs)
  
  # res$status == 0 means feasible solution was found 
  return(res$status == 0)
}
