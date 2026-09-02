# -*- coding: utf-8 -*-
"""
MuleShield AI -- Cash-Withdrawal Location Forecasting
SIH26184 | MHA / I4C

Serves the problem statement's primary ask: forecast likely cash withdrawal
locations in advance.

Two models:
  1. ConditionalLogitRanker -- ranks the K reachable ATMs for a terminal account
  2. XGBRegressor           -- estimates minutes until the withdrawal

Where a cashout happens is a discrete choice among alternatives, and its drivers
compose multiplicatively (proximity x surveillance risk x bank affinity x the
crew's established habits). In log space that product is a sum, which is exactly
a conditional logit. A 953-way softmax over the national ATM directory saw ~5
examples per class and scored below a plain "nearest ATM" rule; a gradient-boosted
ranker had to approximate products with axis-aligned steps and also lost.

An earlier revision documented a "Bayesian Spatial Prior Reranking" step here.
That code was never reachable (it required a label encoder the trainer never
supplied) and has been removed rather than left as an advertised feature that
does not run.

Inference output per complaint:
    {
        "ranked_candidates": [{"atm_id": ..., "confidence": ..., "rank": 1}, ...],
        "search_zone": {"lat": ..., "lon": ..., "radius_km": ..., "atm_count": ...},
        "time_to_cashout_minutes": 23.4,
        "interception_confidence": 0.91,
    }
"""

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_cashout.pkl"

# Hops of the money graph used to approximate an account's crew when counting
# prior cashouts. FeatureBuilder.neighbourhood_prior imports this so training and
# serving expand the same neighbourhood; a mismatch silently shrinks the strongest
# non-distance signal at serving time.
CREW_HOPS = 3

# How many ranked candidate locations the system returns.
#
# SIH26184 asks for withdrawal *locations* to search, not one machine. The
# operating point is 5: measured containment at K=5 is 0.7258 on the held-out
# split (scripts/topk_curve.py), against 0.5621 at K=3, while still cutting the
# search from 1,000 ATMs to 5 -- a 99.5% reduction. K is exposed here rather
# than defaulted at each call site, because the interface contract ("Top 5
# Priority Search Locations") and the number the evaluation reports must not be
# able to drift apart.
OPERATING_K = 5


def _haversine_array(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from one point to an array of points."""
    r = 6371.0
    p1 = math.radians(lat)
    p2 = np.radians(lats)
    dphi = np.radians(lats - lat)
    dlam = np.radians(lons - lon)
    a = np.sin(dphi / 2.0) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))




class ConditionalLogitRanker:
    """
    Conditional-logit ranker over the ATM candidate set.

    Where a cashout happens is a discrete choice among alternatives, and the
    simulator — like the criminology it is drawn from — composes that choice
    multiplicatively: proximity x surveillance risk x bank affinity x the crew's
    established habits. Taking logs turns that product into a sum, which is
    exactly a conditional logit:

        P(atm_i | candidates) = softmax_i( w . log_features_i )

    A gradient-boosted ranker had to approximate those products with axis-aligned
    steps and consistently scored below a plain "nearest ATM" rule. This form
    matches the generating process, trains in under a second, and its weights
    read directly as elasticities an evaluator can sanity-check.
    """

    def __init__(self, n_features: int, k: int):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
            epochs: int = 400, lr: float = 0.5) -> "ConditionalLogitRanker":
        """Maximise the log-likelihood of the chosen ATM within each candidate set."""
        order = np.argsort(groups, kind="stable")
        Xs, ys = X[order].astype(np.float64), y[order]
        n_groups = len(np.unique(groups))
        Xg = Xs.reshape(n_groups, self.k, X.shape[1])
        yg = ys.reshape(n_groups, self.k)

        for _ in range(epochs):
            u = Xg @ self.w
            u -= u.max(axis=1, keepdims=True)
            p = np.exp(u)
            p /= p.sum(axis=1, keepdims=True)
            grad = ((yg - p)[:, :, None] * Xg).sum(axis=(0, 1)) / n_groups
            self.w += lr * grad
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Utility score per candidate row (higher = more likely)."""
        return np.asarray(X, dtype=np.float64) @ self.w


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class MuleXGBPredictor:
    """
    Wraps the ConditionalLogitRanker (where) + XGBRegressor (when).

    Ranking needs the account's coordinates to build its ATM candidate set, and
    the account id to look up prior cashouts by its crew.

    Usage:
        predictor = MuleXGBPredictor.load()
        result = predictor.predict(
            feature_vector_80d,
            node_lat=28.6139, node_lon=77.2090,
            node_bank="HDFC Bank", account="1234-5678-9012",
        )
    """

    def __init__(
        self,
        classifier,         # XGBClassifier
        regressor,          # XGBRegressor
        atm_ids: list[str],
        scaler,             # sklearn StandardScaler
        label_encoder=None, # sklearn LabelEncoder (encoded -> atm_ids index)
        atm_lats: Optional[np.ndarray] = None,   # for candidate ranking
        atm_lons: Optional[np.ndarray] = None,
        atm_risk_scores: Optional[np.ndarray] = None,
        atm_banks: Optional[np.ndarray] = None,
        atm_fraud_counts: Optional[np.ndarray] = None,
        atm_prior_counts: Optional[np.ndarray] = None,
        acct_atm_hist: Optional[dict] = None,
        graph_adj: Optional[dict] = None,
        behaviour: Optional[dict] = None,
        regressor_lo=None,          # 10th-percentile countdown model
        regressor_hi=None,          # 90th-percentile countdown model
        candidate_k: int = 25,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.atm_ids = atm_ids
        self.scaler = scaler
        self.label_encoder = label_encoder  # maps 0..N-1 -> original ATM index
        self.atm_lats = atm_lats
        self.atm_lons = atm_lons
        self.atm_risk_scores = atm_risk_scores
        self.atm_banks = atm_banks
        self.atm_fraud_counts = atm_fraud_counts
        self.atm_prior_counts = atm_prior_counts
        # account -> {atm_index: times cashed out there} over the history window
        self.acct_atm_hist = acct_atm_hist or {}
        # account -> set of graph neighbours, for reaching the account's crew
        self.graph_adj = graph_adj or {}
        # account -> account-behaviour vector (BEHAVIOUR_COLS order), used by
        # the countdown regressor. Inert for ranking, informative for timing.
        self.behaviour = behaviour or {}
        self.regressor_lo = regressor_lo
        self.regressor_hi = regressor_hi
        self.candidate_k = candidate_k


    # Column offsets inside the candidate block (see FeatureBuilder.candidate_block)
    _C_DIST, _C_RISK, _C_SAME, _C_ATMP, _C_CREWP, _C_HAS = 0, 3, 5, 8, 9, 10

    @staticmethod
    def log_features(block: np.ndarray) -> np.ndarray:
        """
        Project the candidate block into the log space the logit operates in.

        Each column is the logarithm of one multiplicative term in the choice
        model, so a linear combination reproduces the product.
        """
        d = block[:, MuleXGBPredictor._C_DIST]
        risk = block[:, MuleXGBPredictor._C_RISK]
        same = block[:, MuleXGBPredictor._C_SAME]
        atmp = block[:, MuleXGBPredictor._C_ATMP]
        crewp = block[:, MuleXGBPredictor._C_CREWP]
        has = block[:, MuleXGBPredictor._C_HAS]
        return np.column_stack([
            -d / 5.0,                  # log distance-decay
            np.log1p(2.0 * risk),      # log surveillance-risk multiplier
            same,                      # log bank-affinity multiplier
            atmp,                      # log global ATM prior
            crewp,                     # log crew prior
            has,                       # crew has used this ATM before at all
        ]).astype(np.float64)

    # ── Serialization ──────────────────────────────────────────────────────────

    def save(self, path: Path = XGB_MODEL_PATH):
        """Save the full predictor bundle to pickle."""
        bundle = {
            "classifier": self.classifier,
            "regressor": self.regressor,
            "atm_ids": self.atm_ids,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "atm_lats": self.atm_lats,
            "atm_lons": self.atm_lons,
            "atm_risk_scores": self.atm_risk_scores,
            "atm_banks": self.atm_banks,
            "atm_fraud_counts": self.atm_fraud_counts,
            "atm_prior_counts": self.atm_prior_counts,
            "acct_atm_hist": self.acct_atm_hist,
            "graph_adj": self.graph_adj,
            "behaviour": self.behaviour,
            "regressor_lo": self.regressor_lo,
            "regressor_hi": self.regressor_hi,
            "candidate_k": self.candidate_k,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path = XGB_MODEL_PATH) -> "MuleXGBPredictor":
        """Load the full predictor bundle from pickle."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(
            classifier=bundle["classifier"],
            regressor=bundle["regressor"],
            atm_ids=bundle["atm_ids"],
            scaler=bundle["scaler"],
            label_encoder=bundle.get("label_encoder"),
            atm_lats=bundle.get("atm_lats"),
            atm_lons=bundle.get("atm_lons"),
            atm_risk_scores=bundle.get("atm_risk_scores"),
            atm_banks=bundle.get("atm_banks"),
            atm_fraud_counts=bundle.get("atm_fraud_counts"),
            atm_prior_counts=bundle.get("atm_prior_counts"),
            acct_atm_hist=bundle.get("acct_atm_hist"),
            graph_adj=bundle.get("graph_adj"),
            behaviour=bundle.get("behaviour"),
            regressor_lo=bundle.get("regressor_lo"),
            regressor_hi=bundle.get("regressor_hi"),
            candidate_k=bundle.get("candidate_k", 25),
        )


    # ── Inference ─────────────────────────────────────────────────────────────

    def _behaviour_dim(self) -> int:
        """Width of the stored behaviour vector (0 if none was bundled)."""
        for v in self.behaviour.values():
            return len(v)
        return 0

    def _crew_prior(self, account: Optional[str], cand_idx: np.ndarray) -> np.ndarray:
        """
        Historical cashouts at each candidate by the account's crew.

        Must expand exactly CREW_HOPS hops — the same depth
        FeatureBuilder.neighbourhood_prior uses when building the training set.
        Serving a shallower neighbourhood than training produces systematically
        smaller counts for the strongest non-distance signal in the model, which
        is a silent train/serve skew rather than an error.
        """
        counts = np.zeros(len(cand_idx), dtype=np.float32)
        if not account or account not in self.graph_adj:
            return counts

        neigh = {account}
        frontier = {account}
        for _ in range(CREW_HOPS):
            nxt = set()
            for n in frontier:
                nxt.update(self.graph_adj.get(n, ()))
            nxt -= neigh
            if not nxt:
                break
            neigh |= nxt
            frontier = nxt

        pos = {int(a): j for j, a in enumerate(cand_idx)}
        for acc in neigh:
            for atm_i, c in self.acct_atm_hist.get(acc, {}).items():
                j = pos.get(atm_i)
                if j is not None:
                    counts[j] += c
        return counts


    # Share of probability mass a search zone must cover. 0.80 keeps the zone
    # tight enough to be worth deploying to while still usually containing the
    # withdrawal; the achieved hit-rate is measured, not assumed.
    ZONE_MASS = 0.80

    def _search_zone(self, cand_idx: np.ndarray, conf: np.ndarray) -> dict:
        """
        Collapse the candidate distribution into one search area.

        SIH26184 asks for withdrawal *locations*, and a patrol is dispatched to an
        area rather than to a single machine. The ranker already produces a
        calibrated distribution over reachable ATMs; this reads it as a spatial
        posterior instead of discarding everything below rank 3.

        The zone is the probability-weighted centroid of the smallest set of
        candidates covering ZONE_MASS of the mass, with a radius that reaches the
        furthest member of that set.
        """
        order = np.argsort(conf)[::-1]
        cum = np.cumsum(conf[order])
        n_keep = int(np.searchsorted(cum, self.ZONE_MASS) + 1)
        n_keep = max(1, min(n_keep, len(order)))
        keep = order[:n_keep]

        idx = cand_idx[keep]
        w = conf[keep]
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))

        lats = self.atm_lats[idx]
        lons = self.atm_lons[idx]
        c_lat = float(np.sum(w * lats))
        c_lon = float(np.sum(w * lons))

        spread = _haversine_array(c_lat, c_lon, lats, lons)
        radius = float(spread.max()) if len(spread) else 0.0

        # How many ATMs of the whole directory fall inside the zone - the number
        # of machines a team would actually have to cover.
        inside = int((_haversine_array(c_lat, c_lon, self.atm_lats, self.atm_lons)
                      <= max(radius, 0.05)).sum())

        return {
            "lat": round(c_lat, 6),
            "lon": round(c_lon, 6),
            "radius_km": round(radius, 3),
            "atm_count": inside,
            "candidates_covered": int(n_keep),
            "probability_mass": round(float(cum[n_keep - 1]), 4),
        }

    def _candidate_block(self, node_lat, node_lon, node_bank="UNKNOWN", account=None):
        """
        Build the K nearest ATM candidates and their ranking features.

        Mirrors FeatureBuilder.candidate_block exactly — the two must stay in
        step or inference features drift away from training features.
        """
        d = _haversine_array(node_lat, node_lon, self.atm_lats, self.atm_lons)
        k = min(self.candidate_k, len(d))
        order = np.argpartition(d, k - 1)[:k]
        order = order[np.argsort(d[order])]

        dk = d[order]
        nearest = max(float(dk[0]), 1e-6)

        risk = (self.atm_risk_scores[order] if self.atm_risk_scores is not None
                else np.zeros(k, dtype=np.float32))
        fraud = (self.atm_fraud_counts[order] if self.atm_fraud_counts is not None
                 else np.zeros(k, dtype=np.float32))
        same_bank = (
            (self.atm_banks[order] == node_bank).astype(np.float32)
            if self.atm_banks is not None else np.zeros(k, dtype=np.float32)
        )
        density = np.array([
            float((_haversine_array(
                float(self.atm_lats[i]), float(self.atm_lons[i]),
                self.atm_lats, self.atm_lons) <= 3.0).sum())
            for i in order
        ], dtype=np.float32)

        atm_prior = (self.atm_prior_counts[order]
                     if self.atm_prior_counts is not None
                     else np.zeros(k, dtype=np.float32))
        crew_prior = self._crew_prior(account, order)

        block = np.column_stack([
            dk.astype(np.float32),
            np.arange(k, dtype=np.float32),
            (dk / nearest).astype(np.float32),
            risk.astype(np.float32),
            np.log1p(fraud).astype(np.float32),
            same_bank,
            density,
            np.exp(-dk / 5.0).astype(np.float32),
            np.log1p(atm_prior).astype(np.float32),
            np.log1p(crew_prior).astype(np.float32),
            (crew_prior > 0).astype(np.float32),
            np.full(k, float((crew_prior > 0).any()), dtype=np.float32),
        ]).astype(np.float32)

        return order, block

    def predict(
        self,
        X: np.ndarray,
        top_k: int = OPERATING_K,
        node_lat: Optional[float] = None,
        node_lon: Optional[float] = None,
        node_bank: str = "UNKNOWN",
        account: Optional[str] = None,
        gnn_mule_prob: float = 0.0,
        chain_timing: Optional[tuple] = None,
    ) -> dict:
        """
        Rank the reachable ATMs for one terminal account and estimate the delay.

        `X` is the 80-dim base vector (64-dim GNN embedding + 16 tabular). The
        K nearest ATMs are scored individually and ranked; confidences are the
        normalised scores across that candidate set.

        This replaces a 953-way softmax over the entire national ATM directory.
        With roughly five training examples per class that model scored below a
        plain "nearest ATM" rule; ranking a local candidate set is learnable and
        matches how the decision is actually made.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[0]

        # Accept the full 80-dim vector for API compatibility and reduce it to
        # the 7-dim ranker context. Must mirror FeatureBuilder.rank_context.
        if X.shape[0] >= 80:
            beh = self.behaviour.get(account)
            if beh is None:
                beh = np.zeros(self._behaviour_dim(), dtype=np.float32)
            X = np.concatenate([
                np.array([
                    X[64], X[65], X[66], X[68], X[70], X[71],
                    float(gnn_mule_prob),
                ], dtype=np.float32),
                np.asarray(beh, dtype=np.float32),
                # Observed hop timing of the traced chain; mirrors
                # FeatureBuilder.rank_context.
                np.log1p(np.asarray(chain_timing or (0.0, 0.0, 0.0),
                                    dtype=np.float32)),
            ])

        if node_lat is None or node_lon is None or self.atm_lats is None:
            raise ValueError(
                "predict() needs node_lat/node_lon to build the ATM candidate set."
            )

        cand_idx, block = self._candidate_block(node_lat, node_lon, node_bank, account)
        rows = np.hstack([
            np.repeat(X[None, :], len(cand_idx), axis=0),
            block,
        ]).astype(np.float32)

        # The logit consumes only the log-space candidate terms; the softmax
        # over the candidate set is the model's own choice probability, so the
        # confidences shown to an operator are calibrated by construction.
        raw = self.classifier.predict(self.log_features(block))
        shifted = raw - float(np.max(raw))
        exp = np.exp(shifted)
        conf = exp / exp.sum() if exp.sum() > 0 else np.full(len(raw), 1.0 / len(raw))
        scores = raw

        rank_order = np.argsort(scores)[::-1][:top_k]
        top = [
            {
                "atm_id": self.atm_ids[int(cand_idx[i])],
                "confidence": round(float(conf[i]), 4),
                "rank": r + 1,
            }
            for r, i in enumerate(rank_order)
        ]

        # The regressor is trained on the base vector plus the winning
        # candidate's block, so the countdown reflects the ATM being dispatched to.
        best_row = self.scaler.transform(rows[int(rank_order[0])].reshape(1, -1))
        minutes = float(self.regressor.predict(best_row)[0])

        # 80% prediction interval, when the quantile models are bundled. A single
        # number hides how much of this delay is genuinely unknowable.
        lo = hi = None
        if self.regressor_lo is not None and self.regressor_hi is not None:
            lo = float(self.regressor_lo.predict(best_row)[0])
            hi = float(self.regressor_hi.predict(best_row)[0])
            lo, hi = max(1.0, min(lo, hi)), max(1.0, max(lo, hi))

        return {
            "ranked_candidates": top,
            "search_zone": self._search_zone(cand_idx, conf),
            "time_to_cashout_minutes": round(max(1.0, minutes), 2),
            "time_to_cashout_low": round(lo, 2) if lo is not None else None,
            "time_to_cashout_high": round(hi, 2) if hi is not None else None,
            "interception_confidence": top[0]["confidence"] if top else 0.0,
        }

    def predict_batch(self, X: np.ndarray, top_k: int = OPERATING_K, **kwargs) -> list[dict]:
        """
        Predict for a batch of feature vectors.

        Each row gets its own ATM candidate set, so this is a loop rather than a
        single vectorised call. Pass per-row coordinates via `node_lats` /
        `node_lons`, or a single shared pair via `node_lat` / `node_lon`.
        """
        rows = X if X.ndim == 2 else X.reshape(1, -1)
        lats = kwargs.pop("node_lats", None)
        lons = kwargs.pop("node_lons", None)
        out = []
        for i, row in enumerate(rows):
            kw = dict(kwargs)
            if lats is not None:
                kw["node_lat"] = float(lats[i])
            if lons is not None:
                kw["node_lon"] = float(lons[i])
            out.append(self.predict(row, top_k=top_k, **kw))
        return out
