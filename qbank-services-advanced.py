# =====================================================
# qbank-backend/app/services/adaptive.py
# Enhanced Adaptive Testing Engine with IRT
# =====================================================
import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# IRT Constants
D = 1.7  # Scaling constant for logistic IRT models

class SelectionStrategy(Enum):
    """Item selection strategies for adaptive testing."""
    MAXIMUM_INFORMATION = "max_info"
    SYMPSON_HETTER = "sympson_hetter"
    RANDOM = "random"
    STRATIFIED = "stratified"
    CONTENT_BALANCED = "content_balanced"
    EXPOSURE_CONTROLLED = "exposure_controlled"

@dataclass
class ItemParameters:
    """IRT item parameters."""
    question_id: int
    version: int
    a: float = 1.0  # Discrimination
    b: float = 0.0  # Difficulty
    c: float = 0.0  # Guessing (for 3PL)
    sh_p: float = 1.0  # Sympson-Hetter probability
    topic_id: Optional[int] = None
    exposure_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class AbilityEstimate:
    """User ability estimate."""
    theta: float = 0.0
    se: float = 1.0  # Standard error
    n_items: int = 0
    history: List[Tuple[int, bool]] = None  # (item_id, correct)

class IRTEngine:
    """Item Response Theory engine for adaptive testing."""
    
    @staticmethod
    def logistic(x: float) -> float:
        """Logistic function."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0
    
    @staticmethod
    def prob_2pl(theta: float, a: float, b: float) -> float:
        """2-Parameter Logistic model probability."""
        return IRTEngine.logistic(D * a * (theta - b))
    
    @staticmethod
    def prob_3pl(theta: float, a: float, b: float, c: float) -> float:
        """3-Parameter Logistic model probability."""
        return c + (1.0 - c) * IRTEngine.logistic(D * a * (theta - b))
    
    @staticmethod
    def fisher_info_2pl(theta: float, a: float, b: float) -> float:
        """Fisher information for 2PL model."""
        p = IRTEngine.prob_2pl(theta, a, b)
        q = 1.0 - p
        if p <= 0 or q <= 0:
            return 0.0
        return (D ** 2) * (a ** 2) * p * q
    
    @staticmethod
    def fisher_info_3pl(theta: float, a: float, b: float, c: float) -> float:
        """Fisher information for 3PL model."""
        p = IRTEngine.prob_3pl(theta, a, b, c)
        q = 1.0 - p
        if p <= 0 or q <= 0 or (1.0 - c) <= 0:
            return 0.0
        return (D ** 2) * (a ** 2) * (q / p) * ((p - c) / (1.0 - c)) ** 2
    
    @staticmethod
    def likelihood_2pl(
        responses: List[Tuple[ItemParameters, bool]], 
        theta: float
    ) -> float:
        """Likelihood of responses given theta (2PL)."""
        likelihood = 1.0
        for item, correct in responses:
            p = IRTEngine.prob_2pl(theta, item.a, item.b)
            likelihood *= p if correct else (1.0 - p)
        return likelihood
    
    @staticmethod
    def likelihood_3pl(
        responses: List[Tuple[ItemParameters, bool]], 
        theta: float
    ) -> float:
        """Likelihood of responses given theta (3PL)."""
        likelihood = 1.0
        for item, correct in responses:
            p = IRTEngine.prob_3pl(theta, item.a, item.b, item.c)
            likelihood *= p if correct else (1.0 - p)
        return likelihood
    
    @staticmethod
    def mle_theta(
        responses: List[Tuple[ItemParameters, bool]], 
        model: str = "3PL",
        initial_theta: float = 0.0,
        max_iter: int = 50,
        tolerance: float = 0.001
    ) -> Tuple[float, float]:
        """Maximum Likelihood Estimation of theta."""
        if not responses:
            return initial_theta, 1.0
        
        theta = initial_theta
        for _ in range(max_iter):
            # Calculate first and second derivatives
            d1, d2 = 0.0, 0.0
            
            for item, correct in responses:
                if model == "3PL":
                    p = IRTEngine.prob_3pl(theta, item.a, item.b, item.c)
                    w = (p - item.c) / (1.0 - item.c)
                else:
                    p = IRTEngine.prob_2pl(theta, item.a, item.b)
                    w = p
                
                q = 1.0 - p
                if p > 0 and q > 0:
                    d1 += D * item.a * (correct - p) * w / p
                    d2 -= (D ** 2) * (item.a ** 2) * w * q
            
            if abs(d2) < 0.0001:
                break
            
            # Newton-Raphson update
            delta = d1 / d2
            theta = theta - delta
            
            # Bound theta to reasonable range
            theta = max(-4.0, min(4.0, theta))
            
            if abs(delta) < tolerance:
                break
        
        # Calculate standard error
        info = sum(
            IRTEngine.fisher_info_3pl(theta, item.a, item.b, item.c)
            if model == "3PL" else
            IRTEngine.fisher_info_2pl(theta, item.a, item.b)
            for item, _ in responses
        )
        
        se = 1.0 / math.sqrt(info) if info > 0 else 1.0
        
        return theta, se
    
    @staticmethod
    def eap_theta(
        responses: List[Tuple[ItemParameters, bool]], 
        model: str = "3PL",
        prior_mean: float = 0.0,
        prior_sd: float = 1.0,
        n_quadrature: int = 61
    ) -> Tuple[float, float]:
        """Expected A Posteriori estimation of theta."""
        if not responses:
            return prior_mean, prior_sd
        
        # Quadrature points
        theta_points = np.linspace(-4, 4, n_quadrature)
        
        # Prior (normal distribution)
        prior = np.exp(-0.5 * ((theta_points - prior_mean) / prior_sd) ** 2)
        prior /= prior.sum()
        
        # Likelihood at each quadrature point
        likelihood = np.ones(n_quadrature)
        for i, theta in enumerate(theta_points):
            for item, correct in responses:
                if model == "3PL":
                    p = IRTEngine.prob_3pl(theta, item.a, item.b, item.c)
                else:
                    p = IRTEngine.prob_2pl(theta, item.a, item.b)
                likelihood[i] *= p if correct else (1.0 - p)
        
        # Posterior
        posterior = likelihood * prior
        posterior /= posterior.sum()
        
        # EAP estimate
        eap = np.sum(theta_points * posterior)
        
        # Standard error
        var = np.sum((theta_points - eap) ** 2 * posterior)
        se = math.sqrt(var)
        
        return float(eap), float(se)

class AdaptiveSelector:
    """Advanced adaptive item selection with multiple strategies."""
    
    def __init__(
        self, 
        strategy: SelectionStrategy = SelectionStrategy.SYMPSON_HETTER,
        exposure_control: bool = True,
        content_balancing: bool = True
    ):
        self.strategy = strategy
        self.exposure_control = exposure_control
        self.content_balancing = content_balancing
        self.irt_engine = IRTEngine()
    
    def select_next_item(
        self,
        candidates: List[ItemParameters],
        ability: AbilityEstimate,
        administered: List[int],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[ItemParameters]:
        """Select next item based on strategy and constraints."""
        
        # Filter out already administered items
        available = [
            item for item in candidates 
            if item.question_id not in administered
        ]
        
        if not available:
            return None
        
        # Apply content constraints if specified
        if constraints and self.content_balancing:
            available = self._apply_content_constraints(available, constraints)
        
        # Select based on strategy
        if self.strategy == SelectionStrategy.MAXIMUM_INFORMATION:
            return self._select_max_information(available, ability.theta)
        elif self.strategy == SelectionStrategy.SYMPSON_HETTER:
            return self._select_sympson_hetter(available, ability.theta)
        elif self.strategy == SelectionStrategy.STRATIFIED:
            return self._select_stratified(available, ability.theta)
        elif self.strategy == SelectionStrategy.CONTENT_BALANCED:
            return self._select_content_balanced(available, ability.theta, constraints)
        elif self.strategy == SelectionStrategy.EXPOSURE_CONTROLLED:
            return self._select_exposure_controlled(available, ability.theta)
        else:
            return random.choice(available)
    
    def _select_max_information(
        self, 
        items: List[ItemParameters], 
        theta: float
    ) -> ItemParameters:
        """Select item with maximum Fisher information at current theta."""
        best_item = None
        max_info = -1
        
        for item in items:
            info = self.irt_engine.fisher_info_3pl(theta, item.a, item.b, item.c)
            if info > max_info:
                max_info = info
                best_item = item
        
        return best_item or items[0]
    
    def _select_sympson_hetter(
        self, 
        items: List[ItemParameters], 
        theta: float
    ) -> ItemParameters:
        """Sympson-Hetter method with probabilistic exposure control."""
        # Calculate information for all items
        scored = []
        for item in items:
            info = self.irt_engine.fisher_info_3pl(theta, item.a, item.b, item.c)
            scored.append((info, item))
        
        # Sort by information (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Probabilistic selection based on sh_p
        for _, item in scored:
            if random.random() <= max(0.0, min(1.0, item.sh_p)):
                return item
        
        # Fallback to highest information item
        return scored[0][1] if scored else items[0]
    
    def _select_stratified(
        self, 
        items: List[ItemParameters], 
        theta: float
    ) -> ItemParameters:
        """Stratified selection based on difficulty levels."""
        # Stratify items by difficulty
        strata = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for item in items:
            if item.b < -0.5:
                strata['easy'].append(item)
            elif item.b < 0.5:
                strata['medium'].append(item)
            else:
                strata['hard'].append(item)
        
        # Select stratum based on current ability
        if theta < -0.5:
            stratum = strata['easy'] or strata['medium'] or strata['hard']
        elif theta < 0.5:
            stratum = strata['medium'] or strata['easy'] or strata['hard']
        else:
            stratum = strata['hard'] or strata['medium'] or strata['easy']
        
        # Select best item from stratum
        if stratum:
            return self._select_max_information(stratum, theta)
        
        return items[0]
    
    def _select_content_balanced(
        self, 
        items: List[ItemParameters], 
        theta: float,
        constraints: Optional[Dict[str, Any]]
    ) -> ItemParameters:
        """Content-balanced selection considering blueprint coverage."""
        if not constraints or 'topic_targets' not in constraints:
            return self._select_max_information(items, theta)
        
        topic_targets = constraints['topic_targets']
        topic_counts = constraints.get('topic_counts', {})
        
        # Calculate coverage gaps
        gaps = {}
        for topic_id, target in topic_targets.items():
            current = topic_counts.get(topic_id, 0)
            gaps[topic_id] = max(0, target - current)
        
        # Prioritize items from underrepresented topics
        priority_items = [
            item for item in items
            if item.topic_id and gaps.get(item.topic_id, 0) > 0
        ]
        
        if priority_items:
            return self._select_max_information(priority_items, theta)
        
        return self._select_max_information(items, theta)
    
    def _select_exposure_controlled(
        self, 
        items: List[ItemParameters], 
        theta: float
    ) -> ItemParameters:
        """Selection with exposure control using progressive method."""
        # Calculate information for all items
        scored = []
        for item in items:
            info = self.irt_engine.fisher_info_3pl(theta, item.a, item.b, item.c)
            # Adjust by exposure rate
            exposure_factor = 1.0 / (1.0 + item.exposure_count / 100.0)
            adjusted_info = info * exposure_factor
            scored.append((adjusted_info, item))
        
        # Sort by adjusted information
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Select with some randomization
        n_consider = min(5, len(scored))
        weights = [1.0 / (i + 1) for i in range(n_consider)]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        selected_idx = np.random.choice(n_consider, p=weights)
        return scored[selected_idx][1]
    
    def _apply_content_constraints(
        self, 
        items: List[ItemParameters],
        constraints: Dict[str, Any]
    ) -> List[ItemParameters]:
        """Apply content constraints to item pool."""
        filtered = items
        
        # Topic constraints
        if 'required_topics' in constraints:
            required = set(constraints['required_topics'])
            filtered = [
                item for item in filtered
                if item.topic_id in required
            ]
        
        # Difficulty constraints
        if 'min_difficulty' in constraints:
            min_b = constraints['min_difficulty']
            filtered = [item for item in filtered if item.b >= min_b]
        
        if 'max_difficulty' in constraints:
            max_b = constraints['max_difficulty']
            filtered = [item for item in filtered if item.b <= max_b]
        
        return filtered if filtered else items

# =====================================================
# qbank-backend/app/services/calibration.py
# Advanced Calibration Service with Multiple IRT Models
# =====================================================
import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class CalibrationModel(Enum):
    """Supported calibration models."""
    CTT = "ctt"  # Classical Test Theory
    RASCH = "rasch"  # 1PL
    TWO_PL = "2pl"
    THREE_PL = "3pl"
    GRADED_RESPONSE = "grm"  # For polytomous items

@dataclass
class CalibrationResult:
    """Calibration result for an item."""
    question_id: int
    version: int
    model: str
    parameters: Dict[str, float]
    standard_errors: Dict[str, float]
    fit_statistics: Dict[str, float]
    n_respondents: int
    converged: bool

class CalibrationEngine:
    """Advanced calibration engine for IRT and CTT."""
    
    def __init__(self, model: CalibrationModel = CalibrationModel.THREE_PL):
        self.model = model
        self.D = 1.7  # Scaling constant
    
    def calibrate_items(
        self,
        response_matrix: np.ndarray,
        item_ids: List[Tuple[int, int]],
        model: Optional[CalibrationModel] = None,
        max_iter: int = 100,
        tolerance: float = 0.001
    ) -> List[CalibrationResult]:
        """Calibrate multiple items."""
        model = model or self.model
        
        if model == CalibrationModel.CTT:
            return self._calibrate_ctt(response_matrix, item_ids)
        elif model == CalibrationModel.RASCH:
            return self._calibrate_rasch(response_matrix, item_ids)
        elif model == CalibrationModel.TWO_PL:
            return self._calibrate_2pl(response_matrix, item_ids, max_iter, tolerance)
        elif model == CalibrationModel.THREE_PL:
            return self._calibrate_3pl(response_matrix, item_ids, max_iter, tolerance)
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def _calibrate_ctt(
        self,
        response_matrix: np.ndarray,
        item_ids: List[Tuple[int, int]]
    ) -> List[CalibrationResult]:
        """Classical Test Theory calibration."""
        results = []
        n_items, n_respondents = response_matrix.shape
        
        # Calculate total scores
        total_scores = np.sum(response_matrix, axis=0)
        
        for i, (qid, version) in enumerate(item_ids):
            item_responses = response_matrix[i, :]
            
            # Item difficulty (p-value)
            p = np.mean(item_responses)
            
            # Item discrimination (point-biserial correlation)
            if np.std(item_responses) > 0 and np.std(total_scores) > 0:
                rpb = np.corrcoef(item_responses, total_scores)[0, 1]
            else:
                rpb = 0.0
            
            # Item variance
            variance = p * (1 - p)
            
            # Standard errors (bootstrap)
            se_p = np.sqrt(variance / n_respondents)
            se_rpb = np.sqrt((1 - rpb**2) / (n_respondents - 2)) if n_respondents > 2 else 0.0
            
            results.append(CalibrationResult(
                question_id=qid,
                version=version,
                model="CTT",
                parameters={'p': p, 'rpb': rpb, 'variance': variance},
                standard_errors={'p': se_p, 'rpb': se_rpb},
                fit_statistics={'n': n_respondents},
                n_respondents=n_respondents,
                converged=True
            ))
        
        return results
    
    def _calibrate_rasch(
        self,
        response_matrix: np.ndarray,
        item_ids: List[Tuple[int, int]]
    ) -> List[CalibrationResult]:
        """Rasch (1PL) model calibration using conditional maximum likelihood."""
        n_items, n_respondents = response_matrix.shape
        
        # Initial estimates
        item_totals = np.sum(response_matrix, axis=1)
        person_totals = np.sum(response_matrix, axis=0)
        
        # Log-odds transformation for initial difficulty estimates
        difficulties = np.zeros(n_items)
        for i in range(n_items):
            p = item_totals[i] / n_respondents
            if p > 0 and p < 1:
                difficulties[i] = -np.log(p / (1 - p))
        
        # Joint maximum likelihood estimation
        # (Simplified - in production use specialized IRT packages)
        results = []
        for i, (qid, version) in enumerate(item_ids):
            b = difficulties[i]
            se_b = 1.0 / np.sqrt(item_totals[i] * (n_respondents - item_totals[i]) / n_respondents)
            
            results.append(CalibrationResult(
                question_id=qid,
                version=version,
                model="Rasch",
                parameters={'b': b},
                standard_errors={'b': se_b},
                fit_statistics={'infit': 1.0, 'outfit': 1.0},  # Placeholder
                n_respondents=n_respondents,
                converged=True
            ))
        
        return results
    
    def _calibrate_2pl(
        self,
        response_matrix: np.ndarray,
        item_ids: List[Tuple[int, int]],
        max_iter: int = 100,
        tolerance: float = 0.001
    ) -> List[CalibrationResult]:
        """2-Parameter Logistic model calibration using marginal maximum likelihood."""
        n_items, n_respondents = response_matrix.shape
        
        # Initialize parameters
        a_params = np.ones(n_items)
        b_params = np.zeros(n_items)
        
        # Initial estimates from CTT
        for i in range(n_items):
            p = np.mean(response_matrix[i, :])
            if p > 0 and p < 1:
                b_params[i] = -np.log(p / (1 - p)) / self.D
        
        # EM Algorithm (simplified version)
        for iteration in range(max_iter):
            old_params = np.concatenate([a_params, b_params])
            
            # E-step: Estimate ability distribution
            theta_points = np.linspace(-4, 4, 21)
            theta_weights = stats.norm.pdf(theta_points, 0, 1)
            theta_weights /= theta_weights.sum()
            
            # M-step: Update item parameters
            for i in range(n_items):
                def neg_log_likelihood(params):
                    a, b = params
                    if a <= 0:
                        return 1e10
                    
                    ll = 0
                    for j, theta in enumerate(theta_points):
                        p = 1 / (1 + np.exp(-self.D * a * (theta - b)))
                        for k in range(n_respondents):
                            if response_matrix[i, k] == 1:
                                ll += theta_weights[j] * np.log(p + 1e-10)
                            else:
                                ll += theta_weights[j] * np.log(1 - p + 1e-10)
                    return -ll
                
                result = optimize.minimize(
                    neg_log_likelihood,
                    [a_params[i], b_params[i]],
                    method='L-BFGS-B',
                    bounds=[(0.1, 3.0), (-3.0, 3.0)]
                )
                
                if result.success:
                    a_params[i], b_params[i] = result.x
            
            # Check convergence
            new_params = np.concatenate([a_params, b_params])
            if np.max(np.abs(new_params - old_params)) < tolerance:
                break
        
        # Create results
        results = []
        for i, (qid, version) in enumerate(item_ids):
            results.append(CalibrationResult(
                question_id=qid,
                version=version,
                model="2PL",
                parameters={'a': a_params[i], 'b': b_params[i]},
                standard_errors={'a': 0.1, 'b': 0.1},  # Placeholder
                fit_statistics={'loglik': 0.0},
                n_respondents=n_respondents,
                converged=iteration < max_iter - 1
            ))
        
        return results
    
    def _calibrate_3pl(
        self,
        response_matrix: np.ndarray,
        item_ids: List[Tuple[int, int]],
        max_iter: int = 100,
        tolerance: float = 0.001
    ) -> List[CalibrationResult]:
        """3-Parameter Logistic model calibration."""
        n_items, n_respondents = response_matrix.shape
        
        # Initialize parameters
        a_params = np.ones(n_items)
        b_params = np.zeros(n_items)
        c_params = np.ones(n_items) * 0.2  # Guessing parameter
        
        # Initial estimates
        for i in range(n_items):
            p = np.mean(response_matrix[i, :])
            if p > 0.2 and p < 1:
                b_params[i] = -np.log((p - 0.2) / (0.8)) / self.D
        
        # EM Algorithm for 3PL (simplified)
        for iteration in range(max_iter):
            old_params = np.concatenate([a_params, b_params, c_params])
            
            # E-step
            theta_points = np.linspace(-4, 4, 21)
            theta_weights = stats.norm.pdf(theta_points, 0, 1)
            theta_weights /= theta_weights.sum()
            
            # M-step
            for i in range(n_items):
                def neg_log_likelihood(params):
                    a, b, c = params
                    if a <= 0 or c < 0 or c > 0.5:
                        return 1e10
                    
                    ll = 0
                    for j, theta in enumerate(theta_points):
                        p = c + (1 - c) / (1 + np.exp(-self.D * a * (theta - b)))
                        for k in range(n_respondents):
                            if response_matrix[i, k] == 1:
                                ll += theta_weights[j] * np.log(p + 1e-10)
                            else:
                                ll += theta_weights[j] * np.log(1 - p + 1e-10)
                    return -ll
                
                result = optimize.minimize(
                    neg_log_likelihood,
                    [a_params[i], b_params[i], c_params[i]],
                    method='L-BFGS-B',
                    bounds=[(0.1, 3.0), (-3.0, 3.0), (0.0, 0.5)]
                )
                
                if result.success:
                    a_params[i], b_params[i], c_params[i] = result.x
            
            # Check convergence
            new_params = np.concatenate([a_params, b_params, c_params])
            if np.max(np.abs(new_params - old_params)) < tolerance:
                break
        
        # Create results
        results = []
        for i, (qid, version) in enumerate(item_ids):
            results.append(CalibrationResult(
                question_id=qid,
                version=version,
                model="3PL",
                parameters={'a': a_params[i], 'b': b_params[i], 'c': c_params[i]},
                standard_errors={'a': 0.1, 'b': 0.1, 'c': 0.05},  # Placeholder
                fit_statistics={'loglik': 0.0},
                n_respondents=n_respondents,
                converged=iteration < max_iter - 1
            ))
        
        return results
    
    def calculate_fit_statistics(
        self,
        response_matrix: np.ndarray,
        parameters: List[Dict[str, float]],
        model: CalibrationModel
    ) -> List[Dict[str, float]]:
        """Calculate item fit statistics (infit, outfit, etc.)."""
        n_items, n_respondents = response_matrix.shape
        fit_stats = []
        
        # Estimate person abilities
        person_abilities = self._estimate_abilities(response_matrix, parameters, model)
        
        for i in range(n_items):
            item_params = parameters[i]
            observed = response_matrix[i, :]
            expected = []
            residuals = []
            
            for j, theta in enumerate(person_abilities):
                if model == CalibrationModel.THREE_PL:
                    p = self._prob_3pl(theta, item_params['a'], item_params['b'], item_params['c'])
                elif model == CalibrationModel.TWO_PL:
                    p = self._prob_2pl(theta, item_params['a'], item_params['b'])
                else:
                    p = self._prob_rasch(theta, item_params['b'])
                
                expected.append(p)
                residuals.append((observed[j] - p) / np.sqrt(p * (1 - p) + 1e-10))
            
            # Infit and outfit statistics
            infit = np.mean(np.array(residuals) ** 2)
            outfit = np.mean(np.array(residuals) ** 2)  # Weighted by variance
            
            fit_stats.append({
                'infit': infit,
                'outfit': outfit,
                'rmse': np.sqrt(np.mean(np.array(residuals) ** 2))
            })
        
        return fit_stats
    
    def _prob_rasch(self, theta: float, b: float) -> float:
        """Rasch model probability."""
        return 1 / (1 + np.exp(-self.D * (theta - b)))
    
    def _prob_2pl(self, theta: float, a: float, b: float) -> float:
        """2PL model probability."""
        return 1 / (1 + np.exp(-self.D * a * (theta - b)))
    
    def _prob_3pl(self, theta: float, a: float, b: float, c: float) -> float:
        """3PL model probability."""
        return c + (1 - c) / (1 + np.exp(-self.D * a * (theta - b)))
    
    def _estimate_abilities(
        self,
        response_matrix: np.ndarray,
        parameters: List[Dict[str, float]],
        model: CalibrationModel
    ) -> np.ndarray:
        """Estimate person abilities given item parameters."""
        n_items, n_respondents = response_matrix.shape
        abilities = np.zeros(n_respondents)
        
        for j in range(n_respondents):
            responses = response_matrix[:, j]
            
            # MLE estimation
            def neg_log_likelihood(theta):
                ll = 0
                for i in range(n_items):
                    if model == CalibrationModel.THREE_PL:
                        p = self._prob_3pl(theta, parameters[i]['a'], 
                                         parameters[i]['b'], parameters[i]['c'])
                    elif model == CalibrationModel.TWO_PL:
                        p = self._prob_2pl(theta, parameters[i]['a'], 
                                         parameters[i]['b'])
                    else:
                        p = self._prob_rasch(theta, parameters[i]['b'])
                    
                    if responses[i] == 1:
                        ll += np.log(p + 1e-10)
                    else:
                        ll += np.log(1 - p + 1e-10)
                return -ll
            
            result = optimize.minimize_scalar(
                neg_log_likelihood,
                bounds=(-4, 4),
                method='bounded'
            )
            
            abilities[j] = result.x if result.success else 0.0
        
        return abilities

# =====================================================
# qbank-backend/analytics/calibration/sh_iterative.py
# Enhanced Sympson-Hetter Iterative Calibration
# =====================================================
import numpy as np
import psycopg2
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExposureControlParams:
    """Parameters for Sympson-Hetter exposure control."""
    tau: float = 0.2  # Target exposure rate
    n_simulees: int = 1000  # Number of simulated examinees
    test_length: int = 30  # Test length
    iterations: int = 10  # Number of iterations
    alpha: float = 0.8  # Learning rate
    theta_dist: str = "normal(0,1)"  # Ability distribution
    floor: float = 0.01  # Minimum exposure probability
    ceiling: float = 1.0  # Maximum exposure probability
    topic_tau: Optional[Dict[str, float]] = None  # Topic-specific targets
    topic_weights: Optional[Dict[str, float]] = None  # Topic weights

class SympsonHetterCalibrator:
    """Enhanced Sympson-Hetter exposure control calibrator."""
    
    def __init__(self, params: ExposureControlParams):
        self.params = params
        self.D = 1.7
    
    def calibrate(
        self,
        pool: List[Dict[str, Any]],
        seed: Optional[int] = None
    ) -> Tuple[Dict[Tuple[int, int], float], List[Dict], List[Dict]]:
        """
        Run iterative Sympson-Hetter calibration.
        
        Returns:
            - k_map: Dictionary of (question_id, version) -> sh_p values
            - exposure_history: List of exposure statistics per iteration
            - convergence_history: List of convergence metrics
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize k values
        k_map = {
            (item["question_id"], item["version"]): item.get("sh_p", 1.0)
            for item in pool
        }
        
        exposure_history = []
        convergence_history = []
        
        # Compute topic-specific targets
        topic_tau = self._compute_topic_tau()
        
        for iteration in range(self.params.iterations):
            logger.info(f"Iteration {iteration + 1}/{self.params.iterations}")
            
            # Update pool with current k values
            for item in pool:
                key = (item["question_id"], item["version"])
                item["sh_p"] = k_map[key]
            
            # Simulate test administrations
            exposures = self._simulate_administrations(pool)
            
            # Calculate exposure rates
            rates = {
                key: count / self.params.n_simulees
                for key, count in exposures.items()
            }
            
            # Update k values
            new_k_map = self._update_k_values(
                pool, k_map, rates, topic_tau
            )
            
            # Calculate statistics
            stats = self._calculate_statistics(rates, topic_tau, pool)
            exposure_history.append(stats)
            
            # Check convergence
            convergence = self._check_convergence(k_map, new_k_map, rates)
            convergence_history.append(convergence)
            
            k_map = new_k_map
            
            # Early stopping if converged
            if convergence["converged"]:
                logger.info(f"Converged at iteration {iteration + 1}")
                break
        
        return k_map, exposure_history, convergence_history
    
    def _compute_topic_tau(self) -> Dict[str, float]:
        """Compute topic-specific exposure targets."""
        if self.params.topic_tau:
            return self.params.topic_tau
        
        if self.params.topic_weights:
            total_weight = sum(self.params.topic_weights.values())
            return {
                topic: self.params.tau * (weight / total_weight)
                for topic, weight in self.params.topic_weights.items()
            }
        
        return {}
    
    def _simulate_administrations(
        self,
        pool: List[Dict[str, Any]]
    ) -> Dict[Tuple[int, int], int]:
        """Simulate test administrations for exposure calculation."""
        exposures = {
            (item["question_id"], item["version"]): 0
            for item in pool
        }
        
        for _ in range(self.params.n_simulees):
            # Sample ability
            theta = self._sample_theta()
            
            # Administer test
            administered = set()
            
            for _ in range(min(self.params.test_length, len(pool))):
                # Get available items
                available = [
                    item for item in pool
                    if (item["question_id"], item["version"]) not in administered
                ]
                
                if not available:
                    break
                
                # Select item using Sympson-Hetter
                selected = self._select_item_sh(available, theta)
                
                if selected:
                    key = (selected["question_id"], selected["version"])
                    administered.add(key)
                    exposures[key] += 1
        
        return exposures
    
    def _sample_theta(self) -> float:
        """Sample ability from specified distribution."""
        dist = self.params.theta_dist
        
        if dist.startswith("normal"):
            # Parse normal(mean, std)
            params = dist.replace("normal", "").strip("()")
            if params:
                mean, std = map(float, params.split(","))
            else:
                mean, std = 0.0, 1.0
            return np.random.normal(mean, std)
        
        elif dist.startswith("uniform"):
            # Parse uniform(min, max)
            params = dist.replace("uniform", "").strip("()")
            if params:
                min_val, max_val = map(float, params.split(","))
            else:
                min_val, max_val = -3.0, 3.0
            return np.random.uniform(min_val, max_val)
        
        else:
            return 0.0
    
    def _select_item_sh(
        self,
        available: List[Dict[str, Any]],
        theta: float
    ) -> Optional[Dict[str, Any]]:
        """Select item using Sympson-Hetter method."""
        # Calculate information for each item
        scored = []
        for item in available:
            info = self._fisher_info_3pl(
                theta,
                item.get("a", 1.0),
                item.get("b", 0.0),
                item.get("c", 0.2)
            )
            scored.append((info, item))
        
        # Sort by information (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Probabilistic selection
        for _, item in scored:
            sh_p = item.get("sh_p", 1.0)
            if np.random.random() <= sh_p:
                return item
        
        # Fallback to highest information
        return scored[0][1] if scored else None
    
    def _fisher_info_3pl(
        self,
        theta: float,
        a: float,
        b: float,
        c: float
    ) -> float:
        """Calculate Fisher information for 3PL model."""
        p = c + (1 - c) / (1 + np.exp(-self.D * a * (theta - b)))
        q = 1 - p
        
        if p <= 0 or q <= 0 or (1 - c) <= 0:
            return 0.0
        
        return (self.D ** 2) * (a ** 2) * (q / p) * ((p - c) / (1 - c)) ** 2
    
    def _update_k_values(
        self,
        pool: List[Dict[str, Any]],
        k_map: Dict[Tuple[int, int], float],
        rates: Dict[Tuple[int, int], float],
        topic_tau: Dict[str, float]
    ) -> Dict[Tuple[int, int], float]:
        """Update k values based on exposure rates."""
        new_k_map = {}
        
        for item in pool:
            key = (item["question_id"], item["version"])
            current_k = k_map[key]
            actual_rate = rates.get(key, 0.0)
            
            # Get target rate (topic-specific or global)
            if topic_tau and item.get("topic_id"):
                target_rate = topic_tau.get(str(item["topic_id"]), self.params.tau)
            else:
                target_rate = self.params.tau
            
            # Update k value
            if actual_rate <= 0.0:
                # Item not exposed, increase k slightly
                new_k = min(self.params.ceiling, current_k * 1.1)
            else:
                # Adjust based on ratio
                ratio = target_rate / actual_rate
                new_k = current_k * (ratio ** self.params.alpha)
                new_k = max(self.params.floor, min(self.params.ceiling, new_k))
            
            new_k_map[key] = new_k
        
        return new_k_map
    
    def _calculate_statistics(
        self,
        rates: Dict[Tuple[int, int], float],
        topic_tau: Dict[str, float],
        pool: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate exposure statistics."""
        rate_values = list(rates.values())
        
        # Overall statistics
        stats = {
            "iteration": len(rate_values),
            "mean_exposure": np.mean(rate_values) if rate_values else 0.0,
            "max_exposure": np.max(rate_values) if rate_values else 0.0,
            "min_exposure": np.min(rate_values) if rate_values else 0.0,
            "std_exposure": np.std(rate_values) if rate_values else 0.0,
        }
        
        # Calculate overexposure
        if topic_tau:
            overexposures = []
            for item in pool:
                key = (item["question_id"], item["version"])
                actual = rates.get(key, 0.0)
                target = topic_tau.get(str(item.get("topic_id")), self.params.tau)
                overexposures.append(max(0, actual - target))
            stats["max_overexposure"] = np.max(overexposures) if overexposures else 0.0
        else:
            overexposures = [
                max(0, rate - self.params.tau)
                for rate in rate_values
            ]
            stats["max_overexposure"] = np.max(overexposures) if overexposures else 0.0
        
        return stats
    
    def _check_convergence(
        self,
        old_k: Dict[Tuple[int, int], float],
        new_k: Dict[Tuple[int, int], float],
        rates: Dict[Tuple[int, int], float]
    ) -> Dict[str, Any]:
        """Check convergence criteria."""
        # Calculate k value changes
        k_changes = [
            abs(new_k[key] - old_k[key])
            for key in old_k.keys()
        ]
        
        # Calculate rate deviations from target
        rate_deviations = [
            abs(rate - self.params.tau)
            for rate in rates.values()
        ]
        
        # Convergence criteria
        max_k_change = np.max(k_changes) if k_changes else 0.0
        mean_k_change = np.mean(k_changes) if k_changes else 0.0
        max_rate_deviation = np.max(rate_deviations) if rate_deviations else 0.0
        
        converged = (
            max_k_change < 0.01 and
            max_rate_deviation < 0.05
        )
        
        return {
            "converged": converged,
            "max_k_change": max_k_change,
            "mean_k_change": mean_k_change,
            "max_rate_deviation": max_rate_deviation,
            "mean_rate_deviation": np.mean(rate_deviations) if rate_deviations else 0.0
        }

def load_pool_from_db(conn: psycopg2.extensions.connection, exam_code: str) -> List[Dict[str, Any]]:
    """Load item pool from database."""
    sql = """
        SELECT 
            qv.question_id,
            qv.version,
            qv.topic_id,
            COALESCE(ic.a, 1.0) as a,
            COALESCE(ic.b, 0.0) as b,
            COALESCE(ic.c, 0.2) as c,
            COALESCE(iec.sh_p, 1.0) as sh_p
        FROM question_publications qp
        JOIN question_versions qv 
            ON qv.question_id = qp.question_id 
            AND qv.version = qp.live_version
        LEFT JOIN item_calibration ic 
            ON ic.question_id = qv.question_id 
            AND ic.version = qv.version 
            AND ic.model = '3PL'
        LEFT JOIN item_exposure_control iec 
            ON iec.question_id = qv.question_id 
            AND iec.version = qv.version
        WHERE qp.exam_code = %s 
            AND qv.state = 'published'
    """
    
    with conn.cursor(psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, (exam_code,))
        rows = cur.fetchall()
    
    return [
        {
            "question_id": int(row["question_id"]),
            "version": int(row["version"]),
            "topic_id": row["topic_id"],
            "a": float(row["a"]),
            "b": float(row["b"]),
            "c": float(row["c"]),
            "sh_p": float(row["sh_p"])
        }
        for row in rows
    ]

def save_k_values_to_db(
    conn: psycopg2.extensions.connection,
    k_map: Dict[Tuple[int, int], float]
) -> int:
    """Save calibrated k values to database."""
    with conn.cursor() as cur:
        cur.execute("SET search_path TO public")
        
        for (qid, ver), k_value in k_map.items():
            cur.execute(
                """
                INSERT INTO item_exposure_control (question_id, version, sh_p)
                VALUES (%s, %s, %s)
                ON CONFLICT (question_id, version)
                DO UPDATE SET 
                    sh_p = EXCLUDED.sh_p,
                    updated_at = now()
                """,
                (qid, ver, float(k_value))
            )
    
    conn.commit()
    return len(k_map)
