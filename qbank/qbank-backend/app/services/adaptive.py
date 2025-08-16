import math, random
from typing import List, Dict, Optional
D = 1.7
def logistic(x: float) -> float: return 1.0 / (1.0 + math.exp(-x))
def prob_3pl(theta: float, a: float, b: float, c: float) -> float:
    return c + (1.0 - c) * logistic(D * a * (theta - b))
def fisher_info_3pl(theta: float, a: float, b: float, c: float) -> float:
    P = prob_3pl(theta, a, b, c); Q = 1.0 - P
    if P<=0 or Q<=0 or (1.0-c)<=0: return 0.0
    return (D**2)*(a**2)*(Q/P)*((P-c)/(1.0-c))**2
def select_vanilla(candidates: List[Dict], theta: float) -> Optional[Dict]:
    best, best_I = None, -1
    for it in candidates:
        I = fisher_info_3pl(theta, it.get("a",1.0), it.get("b",0.0), it.get("c",0.2))
        if I > best_I: best_I, best = I, it
    return best
def select_sympson_hetter(candidates: List[Dict], theta: float) -> Optional[Dict]:
    scored = []
    for it in candidates:
        I = fisher_info_3pl(theta, it.get("a",1.0), it.get("b",0.0), it.get("c",0.2))
        scored.append((I, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    for _, it in scored:
        sh_p = it.get("sh_p", 1.0)
        if random.random() <= max(0.0, min(1.0, sh_p)):
            return it
    return scored[0][1] if scored else None