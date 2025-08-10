# PageRank AI

## Overview
This project implements the PageRank algorithm to rank web pages by importance based on their link structure.

## Features
- **Sampling Method:** Estimates PageRank via a random surfer model.
- **Iterative Method:** Calculates PageRank by iteratively applying the formula until convergence.
- Uses a damping factor (default 0.85) to model random jumps.

```math
PR(p) = \frac{1 - d}{N} + d \sum_{i} \frac{PR(i)}{NumLinks(i)}


