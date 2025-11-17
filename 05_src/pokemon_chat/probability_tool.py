"""
Card draw probability calculations using hypergeometric distribution.
Implements exact probabilities for drawing cards from a deck without replacement.

Mathematical Model:
- Hypergeometric distribution (without replacement)
- Deck size: N (typically 60 for Pokémon TCG)
- Draw size: n (number of cards drawn)
- Target sets: K_A, K_B (number of copies of each card type)

Core Functions (Building Blocks):
1. P(at least n copies of A) - Single hypergeometric: at least n copies of a single card type
2. P(A or B) - OR relationship: at least one from either of two disjoint sets
3. P(A and B) - AND relationship: at least one A AND at least one B

Usage Guide for LLM:
- For simple queries: "probability of drawing at least n [card name]"
  → Use p_at_least_n(deck_size, count, draw_size, min_copies=n)
  → Default min_copies=1 for "at least one"
  
- For OR queries: "probability of drawing [card A] or [card B]"
  → Use p_a_or_b(deck_size, count_a, count_b, draw_size)
  
- For AND queries: "probability of drawing both [card A] AND [card B]"
  → Use p_a_and_b(deck_size, count_a, count_b, draw_size)

- For complex queries: Break down into building blocks
  Example: "P(A or (B and C))" should be calculated as:
    - First: P_B_and_C = p_a_and_b(deck_size, count_b, count_c, draw_size)
    - Then: P_A = p_at_least_n(deck_size, count_a, draw_size, min_copies=1)
    - Finally: P(A or (B and C)) = P_A + P_B_and_C - P(A and (B and C))
    - Where P(A and (B and C)) = p_a_and_b(deck_size, count_a, count_b+count_c, draw_size)
    (assuming B and C are disjoint, so B and C together = count_b + count_c)

Important: All functions assume disjoint sets (no overlap between card types).
"""

from math import comb


def p_at_least_n(deck_size: int, count: int, draw_size: int, min_copies: int = 1) -> float:
    """
    Calculate P(at least n copies of A) - probability of drawing at least min_copies copies of card A.
    
    Single hypergeometric distribution case.
    Uses complement: P(at least n) = 1 - P(less than n)
    
    Formula: P(at least n) = 1 - sum(k=0 to n-1) [C(K, k) * C(N-K, draw_size-k) / C(N, draw_size)]
    where:
        N = deck_size (total cards)
        K = count (copies of card A)
        draw_size = number of cards to draw
        n = min_copies (minimum number of copies to draw)
        C(a, b) = combination of a choose b
    
    Special case (min_copies=1): P(at least 1) = 1 - C(N-K, draw_size) / C(N, draw_size)
    
    Example: "What's the probability of drawing at least one Pikachu in my opening hand?"
        - If deck has 4 Pikachu, deck_size=60, draw_size=7, min_copies=1
        - Use: p_at_least_n(60, 4, 7, 1) or p_at_least_n(60, 4, 7)  # 1 is default
    
    Example: "What's the probability of drawing at least 2 Pikachu?"
        - Use: p_at_least_n(60, 4, 7, 2)
    
    Args:
        deck_size: Total deck size N (typically 60)
        count: Number of copies K of card A in deck
        draw_size: Number of cards to draw
        min_copies: Minimum number of copies to draw (default: 1)
        
    Returns:
        Probability in [0.0, 1.0]
    """
    if count < 0 or draw_size < 0 or deck_size < 0 or min_copies < 0:
        raise ValueError("All parameters must be non-negative")
    if draw_size > deck_size:
        raise ValueError("draw_size cannot exceed deck_size")
    if count > deck_size:
        raise ValueError("count cannot exceed deck_size")
    if min_copies > count:
        return 0.0
    if min_copies > draw_size:
        return 0.0
    
    if count == 0:
        return 0.0
    if draw_size == 0:
        return 0.0 if min_copies > 0 else 1.0
    
    N = deck_size
    K = count
    n = draw_size
    m = min_copies
    
    # Special case: at least 1 (most common)
    if m == 1:
        if N - K < n:
            return 1.0
        prob_none = comb(N - K, n) / comb(N, n)
        return 1.0 - prob_none
    
    # General case: at least m copies
    # P(at least m) = 1 - P(less than m) = 1 - sum(k=0 to m-1) P(exactly k)
    prob_less_than_m = 0.0
    for k in range(m):
        if k > K or (n - k) > (N - K) or (n - k) < 0:
            continue
        if k > n:
            break
        # P(exactly k copies) = C(K, k) * C(N-K, n-k) / C(N, n)
        prob_exactly_k = (comb(K, k) * comb(N - K, n - k)) / comb(N, n)
        prob_less_than_m += prob_exactly_k
    
    return 1.0 - prob_less_than_m


def p_a_or_b(deck_size: int, count_a: int, count_b: int, draw_size: int) -> float:
    """
    Calculate P(A or B) - probability of drawing at least one card from either
    set A or set B (or both).
    
    OR relationship for two disjoint sets (no overlap between card types).
    Uses inclusion-exclusion: P(A or B) = P(A) + P(B) - P(A and B)
    
    Formula: P(A or B) = 1 - C(N - K_A - K_B, n) / C(N, n)
    where we use complement: P(any) = 1 - P(none from either set)
    
    Example: "What's the probability of drawing at least one Pikachu or Charmander?"
        - If deck has 4 Pikachu (A) and 3 Charmander (B), deck_size=60, draw_size=7
        - Use: p_a_or_b(60, 4, 3, 7)
    
    Important: Assumes A and B are disjoint sets (no card belongs to both sets).
    
    Args:
        deck_size: Total deck size N (typically 60)
        count_a: Number of copies of card A (disjoint from B)
        count_b: Number of copies of card B (disjoint from A)
        draw_size: Number of cards to draw n
        
    Returns:
        Probability in [0.0, 1.0]
    """
    if count_a < 0 or count_b < 0 or draw_size < 0 or deck_size < 0:
        raise ValueError("All parameters must be non-negative")
    if draw_size > deck_size:
        raise ValueError("draw_size cannot exceed deck_size")
    if count_a + count_b > deck_size:
        raise ValueError("Sum of counts cannot exceed deck_size")
    
    if count_a == 0 and count_b == 0:
        return 0.0
    if draw_size == 0:
        return 0.0
    
    N = deck_size
    K_A = count_a
    K_B = count_b
    n = draw_size
    
    if N - K_A - K_B < n:
        return 1.0
    
    # P(none from A or B) = P(none from A and none from B)
    prob_none = comb(N - K_A - K_B, n) / comb(N, n)
    return 1.0 - prob_none


def p_a_and_b(deck_size: int, count_a: int, count_b: int, draw_size: int) -> float:
    """
    Calculate P(A and B) - probability of drawing at least one A AND at least one B.
    
    AND relationship using inclusion-exclusion principle.
    
    Formula: P(A and B) = 1 - P(no A) - P(no B) + P(no A and no B)
    where:
        P(no A) = C(N - K_A, n) / C(N, n)
        P(no B) = C(N - K_B, n) / C(N, n)
        P(no A and no B) = C(N - K_A - K_B, n) / C(N, n)
    
    Example: "What's the probability of drawing both an Energy card AND a Supporter card?"
        - If deck has 10 Energy (A) and 8 Supporters (B), deck_size=60, draw_size=7
        - Use: p_a_and_b(60, 10, 8, 7)
    
    Note: Assumes A and B are disjoint sets (no overlap).
    
    Args:
        deck_size: Total deck size N (typically 60)
        count_a: Number of copies of card A (disjoint from B)
        count_b: Number of copies of card B (disjoint from A)
        draw_size: Number of cards to draw n
        
    Returns:
        Probability in [0.0, 1.0]
    """
    if count_a < 0 or count_b < 0 or draw_size < 0 or deck_size < 0:
        raise ValueError("All parameters must be non-negative")
    if draw_size > deck_size:
        raise ValueError("draw_size cannot exceed deck_size")
    if count_a + count_b > deck_size:
        raise ValueError("Sum of counts cannot exceed deck_size")
    
    if count_a == 0 or count_b == 0:
        return 0.0
    if draw_size == 0:
        return 0.0
    
    N = deck_size
    K_A = count_a
    K_B = count_b
    n = draw_size
    
    # P(no A)
    if N - K_A < n:
        prob_no_A = 0.0
    else:
        prob_no_A = comb(N - K_A, n) / comb(N, n)
    
    # P(no B)
    if N - K_B < n:
        prob_no_B = 0.0
    else:
        prob_no_B = comb(N - K_B, n) / comb(N, n)
    
    # P(no A and no B)
    if N - K_A - K_B < n:
        prob_no_A_and_B = 0.0
    else:
        prob_no_A_and_B = comb(N - K_A - K_B, n) / comb(N, n)
    
    return 1.0 - prob_no_A - prob_no_B + prob_no_A_and_B



