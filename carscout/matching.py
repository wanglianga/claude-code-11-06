"""Cross-platform merge engine.

Order of operations for every active listing without a vehicle:
  1. Platform-local dedup is enforced by the DB unique key (platform, listing_id).
  2. VIN rule: identical VIN tail-6 (or full VIN) => merge immediately.
  3. Otherwise recall candidates by brand/series + registration month window +
     mileage band + city, score them (image pHash, description text, numeric
     agreement), apply learned rules, and then:
        score >= AUTO_MERGE_SCORE  -> auto merge
        GRAY_ZONE <= score < auto  -> manual review queue
        score < GRAY_ZONE          -> keep separate (candidate row for audit)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

from carscout.config import settings
from carscout.db import pool
from carscout.utils import (hamming_hex, phash_similarity, price_median,
                            series_similarity, text_similarity)

M = 10000.0


@dataclass
class ScoreBreakdown:
    total: float
    components: dict[str, float]
    hard_conflict: str | None = None

    def to_json(self) -> str:
        return json.dumps({"components": self.components, "hard_conflict": self.hard_conflict},
                          ensure_ascii=False, sort_keys=True)


def _num_similarity(a: float | None, b: float | None, *, tolerance: float, scale: float,
                    max_points: float) -> float:
    """0..max_points based on |a-b|; 0 when one side is missing."""
    if a is None or b is None:
        return 0.0
    diff = abs(float(a) - float(b))
    if diff <= tolerance:
        return max_points
    excess = diff - tolerance
    return max(0.0, max_points * math.exp(-excess / scale))


def _reg_month_index(y: int | None, m: int | None) -> int | None:
    if y is None:
        return None
    return y * 12 + ((m or 6) - 1)


def score_pair(a: dict[str, Any], b: dict[str, Any], phash_a: str | None, phash_b: str | None) -> ScoreBreakdown:
    comps: dict[str, float] = {}
    hard_conflict = None

    # hard identifiers -----------------------------------------------------
    vy_a = _reg_month_index(a.get("reg_year"), a.get("reg_month"))
    vy_b = _reg_month_index(b.get("reg_year"), b.get("reg_month"))
    city_a, city_b = (a.get("city") or "").strip(), (b.get("city") or "").strip()

    # brand/series: hard reject on series mismatch
    ser = series_similarity(
        f"{a.get('brand') or ''}{a.get('series') or ''}", f"{b.get('brand') or ''}{b.get('series') or ''}")
    comps["series"] = round(ser * 0.34, 4)
    if ser < 0.70:
        return ScoreBreakdown(0.0, {"series": comps["series"]}, hard_conflict="series_mismatch")
    comps["series_raw"] = ser

    # VIN tail ---------------------------------------------------------------
    ta, tb = a.get("vin_tail6"), b.get("vin_tail6")
    if ta and tb and ta == tb:
        return ScoreBreakdown(1.0, {"vin_tail6": 1.0}, hard_conflict=None)
    if ta and tb and ta != tb:
        # both platforms published a (different) VIN tail: same car is very unlikely
        comps["vin_tail6"] = 0.0
        comps["vin_conflict"] = -0.10

    # registration month -----------------------------------------------------
    if vy_a is not None and vy_b is not None:
        delta = abs(vy_a - vy_b)
        comps["reg_month"] = round(min(0.16, max(0.0, 0.16 - delta / 10.0)), 4)
        comps["reg_delta_months"] = delta
        if delta > settings.recall_reg_months:
            hard_conflict = "reg_date_conflict"
    else:
        comps["reg_month"] = 0.04  # unknown: small neutral credit

    # mileage ---------------------------------------------------------------
    ma, mb = a.get("mileage_km"), b.get("mileage_km")
    comps["mileage"] = round(_num_similarity(ma, mb, tolerance=1500, scale=12000, max_points=0.14), 4)
    if ma and mb:
        comps["mileage_delta"] = abs(int(ma) - int(mb))

    # city -----------------------------------------------------------------
    comps["city"] = 0.08 if (city_a and city_b and city_a == city_b) else 0.0

    # transmission / displacement / transfers ------------------------------
    if a.get("transmission") and b.get("transmission"):
        comps["transmission"] = 0.04 if str(a["transmission"]) == str(b["transmission"]) else 0.0
    else:
        comps["transmission"] = 0.01
    da, db = a.get("displacement_ml"), b.get("displacement_ml")
    if da is not None and db is not None and da and db:
        comps["displacement"] = round(_num_similarity(da, db, tolerance=50, scale=400,
                                                      max_points=0.03), 4)
    else:
        comps["displacement"] = 0.0
    ta2, tb2 = a.get("transfer_count"), b.get("transfer_count")
    if ta2 is not None and tb2 is not None:
        comps["transfer"] = 0.03 if int(ta2) == int(tb2) else 0.0

    # image pHash ------------------------------------------------------------
    image = None
    if phash_a and phash_b:
        dist = hamming_hex(phash_a, phash_b)
        image = phash_similarity(dist)
        comps["phash_hamming"] = float(dist)
        comps["image"] = round(image * 0.30, 4)
    else:
        comps["image"] = 0.0

    # description text ------------------------------------------------------
    text = None
    if a.get("description") and b.get("description"):
        text = text_similarity(a.get("description"), b.get("description"))
        comps["text_raw"] = text
        comps["text"] = round(text * 0.26, 4)
    else:
        comps["text"] = 0.0

    # Evidence gates. Two independent sources (a photograph of a *different*
    # car, and a description that reads like a different car) must not both be
    # ignored just because the structured fields happen to line up. Any such
    # disagreement caps the pair inside the grey zone so a human decides.
    gate_cap: float | None = None
    if comps.get("phash_hamming") is not None and comps["phash_hamming"] >= 12:
        gate_cap = float(settings.gray_zone_score) + 0.17
        comps["gate_image_conflict"] = 1.0
    if text is not None and text < 0.25:
        cap = float(settings.gray_zone_score) + 0.17
        gate_cap = min(gate_cap, cap) if gate_cap is not None else cap
        comps["gate_text_disagreement"] = 1.0

    # price sanity (informational; never a hard conflict) --------------------
    pa, pb = a.get("price_yuan"), b.get("price_yuan")
    if pa and pb:
        ratio = min(pa, pb) / max(pa, pb)
        comps["price_ratio"] = round(ratio, 4)

    total = sum(v for k, v in comps.items() if k in {
        "series", "reg_month", "mileage", "city", "transmission", "displacement",
        "transfer", "image", "text", "vin_tail6", "vin_conflict"})
    total = max(0.0, min(1.0, round(total, 4)))
    if gate_cap is not None and total > gate_cap:
        comps["capped_by_gate"] = round(total, 4)
        total = round(gate_cap, 4)
    return ScoreBreakdown(total, comps, hard_conflict)


def _phash_of(row: dict[str, Any]) -> str | None:
    """Return the 16-char hex pHash of a row.

    psycopg hands bit(64) columns back as a 64-character '0'/'1' string, so this
    normalises whatever we get into hex. SQL converts back with ('x'||%s)::bit(64).
    """
    return _bitstr_to_hex(row.get("image_phash"))


def phash_literal(hex_value: str | None) -> str | None:
    """SQL fragment helper value: hex digits, to be wrapped as ('x'||%s)::bit(64)."""
    if not hex_value:
        return None
    return format(int(hex_value, 16), "016x")


def _bitstr_to_hex(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if all(c in "01" for c in s) and len(s) in (64, 128):
        return format(int(s, 2), "016x")
    if len(s) == 16 and all(c in "0123456789abcdef" for c in s.lower()):
        return s.lower()
    return None


# ---------------------------------------------------------------------------
# vehicle creation & refresh
# ---------------------------------------------------------------------------
def _vehicle_fields_from(listing: dict[str, Any]) -> dict[str, Any]:
    return {
        "brand": listing.get("brand"), "series": listing.get("series"),
        "reg_year": listing.get("reg_year"), "reg_month": listing.get("reg_month"),
        "mileage_km": listing.get("mileage_km"), "displacement_ml": listing.get("displacement_ml"),
        "transmission": listing.get("transmission"), "transfer_count": listing.get("transfer_count"),
        "city": listing.get("city"), "seller_type": listing.get("seller_type"),
        "win_vin": listing.get("vin"), "vin_tail6": listing.get("vin_tail6"),
        "title_rep": listing.get("title"), "description_rep": listing.get("description"),
        "image_phash": phash_literal(_phash_of(listing)),
        "first_listed_at": listing.get("listed_at"),
        "last_seen_at": listing.get("last_seen_at"),
    }


def create_vehicle(listing: dict[str, Any], confidence: float = 1.0,
                   status: str = "auto") -> int:
    f = _vehicle_fields_from(listing)
    with pool.tx() as conn:
        cur = conn.execute(
            """INSERT INTO vehicles
               (brand, series, reg_year, reg_month, mileage_km, displacement_ml, transmission,
                transfer_count, city, seller_type, win_vin, vin_tail6, title_rep, description_rep,
                image_phash, first_listed_at, last_seen_at, platform_count, merged_confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,('x'||%s::text)::bit(64),%s,%s,0,%s)
               RETURNING id""",
            (f["brand"], f["series"], f["reg_year"], f["reg_month"], f["mileage_km"],
             f["displacement_ml"], f["transmission"], f["transfer_count"], f["city"],
             f["seller_type"], f["win_vin"], f["vin_tail6"], f["title_rep"], f["description_rep"],
             f["image_phash"], f["first_listed_at"], f["last_seen_at"], confidence),
        )
        vid = int(cur.fetchone()["id"])
        cur.close()
        conn.execute("UPDATE listings SET vehicle_id = %s, match_status = %s, updated_at = now() WHERE id = %s",
                     (vid, status, listing["id"]))
        _refresh_vehicle_stats(conn, vid)
        return vid


def attach_to_vehicle(listing_id: int, vehicle_id: int, *, confidence: float, status: str = "auto") -> None:
    with pool.tx() as conn:
        conn.execute(
            """UPDATE listings SET vehicle_id = %s, match_status = %s, match_score = %s, updated_at = now()
               WHERE id = %s""", (vehicle_id, status, confidence, listing_id))
        _refresh_vehicle_stats(conn, vehicle_id)


def detach_from_vehicle(listing_id: int) -> int | None:
    with pool.tx() as conn:
        cur = conn.execute("SELECT vehicle_id FROM listings WHERE id = %s", (listing_id,))
        row = cur.fetchone()
        cur.close()
        vid = row.get("vehicle_id") if row else None
        conn.execute("""UPDATE listings SET vehicle_id = NULL, match_status = 'unmatched',
                        updated_at = now() WHERE id = %s""", (listing_id,))
        if vid:
            _refresh_vehicle_stats(conn, vid)
        return vid


def _refresh_vehicle_stats(conn, vehicle_id: int) -> None:
    cur = conn.execute(
        """SELECT count(DISTINCT platform) AS platforms, count(*) AS n,
                  bool_and(is_active) AS all_active,
                  min(first_seen_at) AS first_seen, max(last_seen_at) AS last_seen
           FROM listings WHERE vehicle_id = %s""", (vehicle_id,))
    stats = cur.fetchone()
    cur.close()
    if not stats or not stats["n"]:
        return
    # representative fields: most recently seen listing wins for display text
    cur = conn.execute(
        """SELECT title, description, city, mileage_km, brand, series, reg_year, reg_month,
                  vin, vin_tail6, image_phash, listed_at, seller_type, transmission,
                  transfer_count, displacement_ml, price_yuan
           FROM listings WHERE vehicle_id = %s
           ORDER BY is_active DESC, last_seen_at DESC, id DESC LIMIT 1""", (vehicle_id,))
    rep = cur.fetchone()
    cur.close()
    ph = phash_literal(_phash_of(rep)) if rep else None
    # price range across platforms (informative for exports)
    cur = conn.execute(
        """SELECT min(price_yuan) AS lo, max(price_yuan) AS hi, count(DISTINCT price_yuan) AS distinct_prices
           FROM listings WHERE vehicle_id = %s AND is_active""", (vehicle_id,))
    pr = cur.fetchone()
    cur.close()
    conn.execute(
        """UPDATE vehicles SET
             brand = COALESCE(%s, brand), series = COALESCE(%s, series),
             reg_year = COALESCE(%s, reg_year), reg_month = COALESCE(%s, reg_month),
             mileage_km = COALESCE(%s, mileage_km), city = COALESCE(%s, city),
             title_rep = COALESCE(%s, title_rep), description_rep = COALESCE(%s, description_rep),
             win_vin = COALESCE(%s, win_vin), vin_tail6 = COALESCE(%s, vin_tail6),
             seller_type = COALESCE(%s, seller_type), transmission = COALESCE(%s, transmission),
             transfer_count = COALESCE(%s, transfer_count),
             displacement_ml = COALESCE(%s, displacement_ml),
             image_phash = COALESCE(('x'||%s::text)::bit(64), image_phash),
             first_listed_at = LEAST(COALESCE(first_listed_at, %s), %s),
             last_seen_at = GREATEST(COALESCE(last_seen_at, %s), %s),
             platform_count = %s, updated_at = now()
           WHERE id = %s""",
        (rep["brand"], rep["series"], rep["reg_year"], rep["reg_month"], rep["mileage_km"],
         rep["city"], rep["title"], rep["description"], rep["vin"], rep["vin_tail6"],
         rep["seller_type"], rep["transmission"], rep["transfer_count"], rep["displacement_ml"],
         ph, rep["listed_at"], rep["listed_at"], stats["last_seen"], stats["last_seen"],
         stats["platforms"], vehicle_id),
    )
    # orphaned single-listing vehicles are still valid (one platform, one record)
    cur = conn.execute("SELECT count(*) AS n FROM listings WHERE vehicle_id = %s", (vehicle_id,))
    n = cur.fetchone()["n"]
    cur.close()
    if n == 0:
        conn.execute("DELETE FROM vehicles WHERE id = %s", (vehicle_id,))


# ---------------------------------------------------------------------------
# candidate recall
# ---------------------------------------------------------------------------
def recall_candidates(listing: dict[str, Any]) -> list[dict[str, Any]]:
    """Recall vehicles (as pseudo-listing bundles) from other platforms."""
    reg_idx = _reg_month_index(listing.get("reg_year"), listing.get("reg_month"))
    rows = pool.query(
        """
        SELECT v.id AS vehicle_id, v.brand, v.series, v.reg_year, v.reg_month, v.mileage_km,
               v.city, v.transmission, v.displacement_ml, v.transfer_count, v.vin_tail6,
               v.description_rep, v.title_rep, v.image_phash,
               (SELECT l.price_yuan FROM listings l WHERE l.vehicle_id = v.id AND l.is_active
                ORDER BY l.last_seen_at DESC LIMIT 1) AS price_yuan,
               (SELECT array_agg(DISTINCT l.platform) FROM listings l WHERE l.vehicle_id = v.id) AS platforms
        FROM vehicles v
        WHERE v.id <> COALESCE(%s, 0)
          AND (%s::int IS NULL OR v.reg_year IS NULL
               OR abs((v.reg_year * 12 + COALESCE(v.reg_month, 6)) - %s::int) <= %s::int)
          AND (%s::text IS NULL OR v.series IS NULL OR v.series = %s::text OR v.series LIKE %s::text)
        ORDER BY v.updated_at DESC
        LIMIT %s::int
        """,
        (listing.get("vehicle_id"),
         reg_idx, reg_idx, settings.recall_reg_months,
         listing.get("series"), listing.get("series"), f"%{listing.get('series') or ''}%",
         settings.candidate_limit),
    )
    out = []
    for r in rows:
        r = dict(r)
        r["description"] = r.pop("description_rep")
        r["title"] = r.pop("title_rep")
        r["image_phash"] = _phash_of(r)
        out.append(r)
    return out


def _recall_is_viable(listing: dict[str, Any], cand: dict[str, Any]) -> bool:
    if listing.get("mileage_km") and cand.get("mileage_km"):
        if abs(int(listing["mileage_km"]) - int(cand["mileage_km"])) > settings.recall_mileage_km:
            return False
    if listing.get("city") and cand.get("city") and listing["city"] != cand["city"]:
        # cross-city listings are possible but only keep them as candidates
        pass
    return True


# ---------------------------------------------------------------------------
# learned rules
# ---------------------------------------------------------------------------
def rule_key_for(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, str]:
    """Symmetric rule key from the pair's discriminative fields."""
    keys = []
    for k in ("brand", "series", "city"):
        va, vb = a.get(k), b.get(k)
        parts = sorted([str(va or ""), str(vb or "")])
        keys.append("=".join(parts) if parts[0] != parts[1] else str(va or ""))
    return "|".join(keys), f"{a.get('series')}|{b.get('series')}"


def lookup_rule(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    key1, key2 = rule_key_for(a, b)
    row = pool.query_one(
        """SELECT * FROM match_rules WHERE active AND rule_key IN (%s,%s) ORDER BY confidence DESC LIMIT 1""",
        (key1, key2))
    return row


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def run_merge_for_listing(listing: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Match a single listing against the vehicle pool. Returns a summary dict."""
    if listing.get("vehicle_id"):
        return {"action": "already_matched", "vehicle_id": listing["vehicle_id"]}
    cands = recall_candidates(listing)
    if not cands:
        vid = create_vehicle(listing, confidence=1.0)
        return {"action": "created_vehicle", "vehicle_id": vid}

    best: tuple[float, dict[str, Any], ScoreBreakdown] | None = None
    for cand in cands:
        if not _recall_is_viable(listing, cand):
            continue
        sb = score_pair(listing, cand, _phash_of(listing), cand.get("image_phash"))
        if sb.hard_conflict == "series_mismatch":
            continue
        if best is None or sb.total > best[0]:
            best = (sb.total, cand, sb)

    if best is None:
        vid = create_vehicle(listing, confidence=1.0)
        return {"action": "created_vehicle", "vehicle_id": vid}

    score, cand, sb = best
    vid = int(cand["vehicle_id"])
    rule = lookup_rule(listing, cand)
    if rule and rule.get("decision") == "merge" and sb.hard_conflict is None:
        _record_candidate(listing["id"], _candidate_listing_id(vid), score, sb, "rule_applied",
                          run_id=run_id, rule_id=rule["id"], note=f"rule {rule['rule_key']}")
        attach_to_vehicle(listing["id"], vid, confidence=score, status="ruled")
        pool.execute("UPDATE match_rules SET applied_count = applied_count + 1, updated_at = now() WHERE id = %s",
                     (rule["id"],))
        return {"action": "merged_by_rule", "vehicle_id": vid, "score": score}
    if rule and rule.get("decision") == "split":
        _record_candidate(listing["id"], _candidate_listing_id(vid), score, sb, "rule_applied",
                          run_id=run_id, rule_id=rule["id"], note=f"rule {rule['rule_key']}")
        new_vid = create_vehicle(listing, confidence=1.0)
        return {"action": "split_by_rule", "vehicle_id": new_vid, "score": score}

    if score >= float(settings.auto_merge_score) and sb.hard_conflict is None:
        _record_candidate(listing["id"], _candidate_listing_id(vid), score, sb, "auto_merged", run_id=run_id)
        attach_to_vehicle(listing["id"], vid, confidence=score)
        return {"action": "merged", "vehicle_id": vid, "score": score}

    if score >= float(settings.gray_zone_score):
        # Grey zone: the pair must not be joined automatically, but leaving the
        # listing fully unattached would make it invisible as a candidate for
        # later listings (two identical unmatched rows could never meet).
        # So: give it its OWN provisional vehicle (marked manual_pending, low
        # confidence) and queue the candidate. A human decision either folds
        # the two vehicles together or confirms they stay apart.
        other_id = _candidate_listing_id(vid)
        cid = _record_candidate(listing["id"], other_id, score, sb, "manual_pending",
                                run_id=run_id)
        # Only the newly arriving listing is parked: it gets its own provisional
        # vehicle. The counterpart keeps whatever cluster it already belongs to;
        # the review row names both listing ids, so a human sees the whole pair.
        own_vid = create_vehicle(listing, confidence=score, status="manual_pending")
        pool.execute("""UPDATE listings SET match_status = 'manual_pending', match_score = %s,
                        updated_at = now() WHERE id = %s""", (score, listing["id"]))
        return {"action": "queued_for_review", "vehicle_id": own_vid,
                "candidate_vehicle_id": vid, "score": score, "candidate_id": cid}

    _record_candidate(listing["id"], _candidate_listing_id(vid), score, sb, "rejected", run_id=run_id)
    new_vid = create_vehicle(listing, confidence=1.0)
    return {"action": "created_vehicle", "vehicle_id": new_vid, "score": score}


def _candidate_listing_id(vehicle_id: int) -> int:
    row = pool.query_one(
        """SELECT id FROM listings WHERE vehicle_id = %s ORDER BY is_active DESC, last_seen_at DESC LIMIT 1""",
        (vehicle_id,))
    return int(row["id"]) if row else 0


def _record_candidate(a: int, b: int, score: float, sb: ScoreBreakdown, state: str, *,
                      run_id: str | None = None, rule_id: int | None = None,
                      note: str | None = None) -> int:
    lo, hi = min(a, b), max(a, b)
    with pool.tx() as conn:
        cur = conn.execute(
            """INSERT INTO match_candidates
               (run_id, listing_a, listing_b, score, components, state, rule_id, note, created_at)
               VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
               ON CONFLICT (listing_a, listing_b) DO UPDATE
                 SET score = EXCLUDED.score, components = EXCLUDED.components,
                     state = EXCLUDED.state, note = EXCLUDED.note
               RETURNING id""",
            (run_id, lo, hi, score, sb.to_json(), state, rule_id, note),
        )
        cid = int(cur.fetchone()["id"])
        cur.close()
        if state == "manual_pending":
            batch = run_id[:8] if run_id else "manual"
            conn.execute(
                """INSERT INTO manual_review_queue (candidate_id, batch_id)
                   VALUES (%s, %s) ON CONFLICT (candidate_id, batch_id) DO NOTHING""",
                (cid, batch),
            )
        return cid


def merge_pending_listings(*, run_id: str | None = None, platforms: Iterable[str] | None = None,
                           limit: int = 500) -> dict[str, Any]:
    """Batch pass: match every unmatched active listing."""
    plat = list(platforms) if platforms else None
    rows = pool.query(
        """SELECT * FROM listings
           WHERE is_active AND (vehicle_id IS NULL OR match_status = 'unmatched')
             AND detail_status IN ('ok','skipped_unchanged')
             AND (%s::text[] IS NULL OR platform = ANY(%s::text[]))
           ORDER BY first_seen_at ASC, id ASC LIMIT %s""",
        (plat, plat, limit))
    summary: dict[str, int] = {}
    for row in rows:
        res = run_merge_for_listing(row, run_id=run_id)
        summary[res["action"]] = summary.get(res["action"], 0) + 1
    return summary


def merge_vehicles(target_id: int, source_id: int, *, confidence: float,
                   status: str = "auto") -> None:
    """Fold `source` vehicle's listings into `target` and delete the empty shell."""
    if target_id == source_id:
        return
    with pool.tx() as conn:
        conn.execute(
            """UPDATE listings SET vehicle_id = %s, match_status = %s,
                      match_score = COALESCE(match_score, %s), updated_at = now()
               WHERE vehicle_id = %s""", (target_id, status, confidence, source_id))
        conn.execute("DELETE FROM vehicles WHERE id = %s AND NOT EXISTS "
                     "(SELECT 1 FROM listings WHERE vehicle_id = %s)", (source_id, source_id))
        _refresh_vehicle_stats(conn, target_id)


def confirm_review(candidate_id: int, decision: str, *, reviewer: str = "operator",
                   note: str | None = None, learn_rule: bool = True) -> dict[str, Any]:
    """Apply a human decision to a grey-zone pair and persist it as a rule."""
    cand = pool.query_one("SELECT * FROM match_candidates WHERE id = %s", (candidate_id,))
    if not cand:
        return {"ok": False, "error": "candidate not found"}
    a = pool.query_one("SELECT * FROM listings WHERE id = %s", (cand["listing_a"],)) or {}
    b = pool.query_one("SELECT * FROM listings WHERE id = %s", (cand["listing_b"],)) or {}
    if not a or not b:
        return {"ok": False, "error": "listing row missing"}

    va, vb = a.get("vehicle_id"), b.get("vehicle_id")
    if decision == "merge":
        if not va and not vb:
            target = create_vehicle(a)
            vb = None
        elif va and not vb:
            target = va
        elif vb and not va:
            target = vb
        else:
            target = va
        for lid in (a["id"], b["id"]):
            attach_to_vehicle(int(lid), int(target), confidence=float(cand["score"]),
                              status="manual_confirmed")
        if vb and int(vb) != int(target):
            merge_vehicles(int(target), int(vb), confidence=float(cand["score"]),
                           status="manual_confirmed")
        state = "manual_confirmed"
    else:
        state = "rejected"
        if va and vb and int(va) == int(vb):
            # same vehicle: detach b and give it its own shell
            detach_from_vehicle(int(b["id"]))
            nb = pool.query_one("SELECT * FROM listings WHERE id = %s", (b["id"],)) or b
            create_vehicle(nb, confidence=0.0, status="rejected")
        # different vehicles already: confirm they stay apart

    rule_id = None
    if learn_rule:
        key = rule_key_for(a, b)[1]
        with pool.tx() as conn:
            cur = conn.execute(
                """INSERT INTO match_rules (rule_key, decision, source, confidence, note)
                   VALUES (%s, %s, 'manual', 0.95, %s)
                   ON CONFLICT (rule_key) DO UPDATE
                     SET decision = EXCLUDED.decision, note = EXCLUDED.note,
                         active = TRUE, updated_at = now()
                   RETURNING id""",
                (key, decision, note or f"learned from candidate {candidate_id}"),
            )
            rule_id = int(cur.fetchone()["id"])
            cur.close()
    pool.execute(
        """UPDATE match_candidates SET state = %s, decided_at = now(), decided_by = %s,
           note = %s, rule_id = %s WHERE id = %s""",
        (state, reviewer, note, rule_id, candidate_id))
    pool.execute(
        """UPDATE manual_review_queue SET status = %s, reviewed_at = now(), reviewer = %s,
           note = %s WHERE candidate_id = %s AND status = 'pending'""",
        ("confirmed_merge" if decision == "merge" else "confirmed_split", reviewer, note, candidate_id))
    return {"ok": True, "state": state, "rule_key": rule_key_for(a, b)[1], "rule_id": rule_id}


def rebuild_all_vehicles(*, keep_rules: bool = True) -> dict[str, Any]:
    """Re-cluster every active listing from scratch.

    Learned rules are kept (and therefore re-applied), so a rebuild reproduces
    earlier human decisions without re-asking. Candidate/queue audit rows are
    rebuilt as well; use this after threshold changes, not routinely.
    """
    rebuild_scope = "all"
    with pool.tx() as conn:
        conn.execute("UPDATE listings SET vehicle_id = NULL, match_status = 'unmatched' WHERE is_active")
        conn.execute("DELETE FROM match_candidates")
        conn.execute("DELETE FROM manual_review_queue")
        conn.execute("DELETE FROM vehicles")
        if not keep_rules:
            conn.execute("DELETE FROM match_rules")
    summary = merge_pending_listings(limit=100000)
    summary["rebuilt_from"] = rebuild_scope
    return summary
