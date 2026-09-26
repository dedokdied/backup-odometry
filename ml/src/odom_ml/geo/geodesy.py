from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = WGS84_E2 / (1.0 - WGS84_E2)


def utm_zone(lon: np.ndarray | float) -> np.ndarray:
    return np.floor((np.asarray(lon, dtype=np.float64) + 180.0) / 6.0).astype(np.int64) + 1


def dominant_utm_zone(lon: np.ndarray | float) -> int:
    """Most frequent UTM zone, ties broken by the lowest zone number.

    The mean of the per-point zone is meaningless whenever the input spans a
    zone boundary, so pick the mode instead.
    """
    z = utm_zone(lon).ravel()
    if z.size == 0:
        raise ValueError("cannot derive a UTM zone from an empty longitude array")
    vals, counts = np.unique(z, return_counts=True)
    return int(vals[np.argmax(counts)])


def latlon_to_utm(
    lat: np.ndarray,
    lon: np.ndarray,
    zone: int | None = None,
    northern: bool | None = None,
    false_northing: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krüger-series transverse Mercator -> UTM.

    ``zone`` defaults to the per-point UTM zone, so arrays that straddle a zone
    boundary stay correct.  Pass an ``int`` to force a single zone (useful for
    building one continuous local frame).

    ``false_northing`` adds the standard 10 000 km offset on the northern
    hemisphere.  Set it to ``False`` to obtain the projection without the
    offset, which is what local map frames are usually built on.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if zone is None:
        zone_arr = utm_zone(lon)
    else:
        zone_arr = np.full(lat.shape, int(zone), dtype=np.int64)
    if northern is None:
        north = lat >= 0.0
    else:
        north = np.full(lat.shape, bool(northern))

    lon0 = np.deg2rad((zone_arr - 1) * 6 - 180 + 3)
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon) - lon0

    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = WGS84_EP2 * np.cos(phi) ** 2
    aa = np.cos(phi) * lam
    m = WGS84_A * (
        (1 - WGS84_E2 / 4 - 3 * WGS84_E2**2 / 64 - 5 * WGS84_E2**3 / 256) * phi
        - (3 * WGS84_E2 / 8 + 3 * WGS84_E2**2 / 32 + 45 * WGS84_E2**3 / 1024) * np.sin(2 * phi)
        + (15 * WGS84_E2**2 / 256 + 45 * WGS84_E2**3 / 1024) * np.sin(4 * phi)
        - (35 * WGS84_E2**3 / 3072) * np.sin(6 * phi)
    )

    easting = (
        500000.0
        + n * (
            aa
            + (1 - t + c) * aa**3 / 6
            + (5 - 18 * t + t**2 + 72 * c - 58 * WGS84_EP2) * aa**5 / 120
        )
    )
    northing = m + n * np.tan(phi) * (
        aa**2 / 2 + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24 + (61 - 58 * t + t**2 + 600 * c - 330 * WGS84_EP2) * aa**6 / 720
    )
    if false_northing:
        northing = northing + np.where(north, 10000000.0, 0.0)
    return easting, northing, zone_arr


def geodetic_to_ecef(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(np.asarray(lat, dtype=np.float64))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float64))
    h = np.asarray(alt, dtype=np.float64)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(phi) ** 2)
    return np.column_stack(
        [
            (n + h) * np.cos(phi) * np.cos(lam),
            (n + h) * np.cos(phi) * np.sin(lam),
            (n * (1 - WGS84_E2) + h) * np.sin(phi),
        ]
    )


def _enu_basis(lat0: float, lon0: float) -> np.ndarray:
    """Rows are the local east/north/up unit vectors in ECEF components."""
    phi0 = np.deg2rad(lat0)
    lam0 = np.deg2rad(lon0)
    return np.array(
        [
            [-np.sin(lam0), np.cos(lam0), 0.0],
            [-np.sin(phi0) * np.cos(lam0), -np.sin(phi0) * np.sin(lam0), np.cos(phi0)],
            [np.cos(phi0) * np.cos(lam0), np.cos(phi0) * np.sin(lam0), np.sin(phi0)],
        ]
    )


def ecef_to_enu(xyz: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    rot = _enu_basis(lat0, lon0)
    origin = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([alt0]))[0]
    return (np.asarray(xyz, dtype=np.float64) - origin) @ rot.T


def ecef_to_geodetic(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF -> geodetic latitude/longitude/altitude (iterative Bowring method)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    # Bowring's starting point, then a few fixed-point refinements.
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(6):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return np.rad2deg(lat), np.rad2deg(lon), alt


def enu_to_ecef(enu: np.ndarray, lat0: float, lon0: float, alt0: float) -> np.ndarray:
    rot = _enu_basis(lat0, lon0)
    origin = geodetic_to_ecef(np.array([lat0]), np.array([lon0]), np.array([alt0]))[0]
    return np.asarray(enu, dtype=np.float64) @ rot + origin


def local_enu_to_latlon(
    enu: np.ndarray, lat0: float, lon0: float, alt0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`latlon_to_local_enu`, for cached ENU trajectories."""
    return ecef_to_geodetic(enu_to_ecef(enu, lat0, lon0, alt0))


def enu_to_utm(
    enu: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float,
    zone: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a cached local-ENU track into UTM.

    The cached labels are stored in a true-north ENU frame, but the jury frame is
    UTM grid axes, so a re-projection (not just an offset) is required.
    """
    lat, lon, alt = local_enu_to_latlon(enu, lat0, lon0, alt0)
    return latlon_to_utm(lat, lon, zone=zone, false_northing=False)


def latlon_to_local_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    lat0: float,
    lon0: float,
    alt0: float,
) -> np.ndarray:
    return ecef_to_enu(geodetic_to_ecef(lat, lon, alt), lat0, lon0, alt0)


# --- MGRS ---------------------------------------------------------------------
# MGRS 100 km square alphabets (the letters I and O are skipped, per the standard).
_MGRS_COL = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ"))
_MGRS_ROW = np.array(list("ABCDEFGHJKLMNPQRSTUV"))


def utm_to_mgrs(
    easting: np.ndarray,
    northing: np.ndarray,
    zone: np.ndarray | int,
    digits: int = 5,
    northern: bool | None = None,
) -> dict[str, np.ndarray | str]:
    """Convert UTM easting/northing to an MGRS grid reference.

    Returns the 100 km square letters plus easting/northing truncated to
    ``digits`` metres *inside* the square.  ``digits=5`` gives the full MGRS
    grid reference (e.g. ``37DB 00990 87980``); ``digits`` is clamped to [2, 5].

    Follows the AA (MGRS-New, WGS84) row-letter scheme: the 100 km row just
    north of the equator is ``A`` in odd-numbered zones and ``F`` in even ones.
    """
    easting = np.asarray(easting, dtype=np.float64)
    northing = np.asarray(northing, dtype=np.float64)
    if np.isscalar(zone):
        zone_arr = np.full(easting.shape, int(zone), dtype=np.int64)
    else:
        zone_arr = np.asarray(zone, dtype=np.int64)
    digits = int(np.clip(digits, 2, 5))
    prec = 10 ** (5 - digits)

    if northern is None:
        # UTM northing >= 5 000 km is northern; below that it is southern
        # (the false northing is 10 000 km, the equator sits at 0 or 10 000 km).
        north = np.mean(northing) >= 5000000.0
    north = np.full(easting.shape, bool(northern)) if np.isscalar(northern) else northern

    col_band = np.floor(easting / 100000.0).astype(np.int64) - 1  # 0 at the 100-200 km band
    col_letter = _MGRS_COL[(col_band + 8 * ((zone_arr - 1) % 3)) % 24]

    # Distance north of the equator, then the row band with the 2 000 km wrap.
    n_from_eq = np.where(north, northing - 10000000.0, northing)
    band = np.floor(n_from_eq / 100000.0).astype(np.int64)
    row_idx = (band + 5 * (1 - zone_arr % 2)) % 20
    row_letter = _MGRS_ROW[row_idx]

    e_t = np.floor((easting - (col_band + 1) * 100000.0) / prec).astype(np.int64)
    n_t = np.floor(np.mod(n_from_eq, 100000.0) / prec).astype(np.int64)
    zone_s = np.array([f"{z:02d}" for z in zone_arr.ravel()])
    e_s = np.array([f"{v:0{digits}d}" for v in e_t.ravel()])
    n_s = np.array([f"{v:0{digits}d}" for v in n_t.ravel()])
    text = np.array(
        [f"{a}{b}{r} {c} {d}" for a, b, r, c, d in zip(zone_s, col_letter, row_letter, e_s, n_s)]
    )
    shape = np.broadcast(easting, northing).shape
    return {
        "zone": zone_arr,
        "col_letter": col_letter,
        "row_letter": row_letter,
        "easting_in_square": e_t,
        "northing_in_square": n_t,
        "easting_m": e_t * prec,
        "northing_m": n_t * prec,
        "text": text.reshape(shape),
    }


def latlon_to_mgrs(lat: np.ndarray, lon: np.ndarray, digits: int = 5) -> dict[str, np.ndarray | str]:
    e, n, z = latlon_to_utm(lat, lon)
    out = utm_to_mgrs(e, n, z, digits=digits, northern=np.asarray(lat) >= 0.0)
    out["easting_utm"] = e
    out["northing_utm"] = n
    return out


def mgrs_to_utm(
    zone: int,
    col_letter: str,
    row_letter: str,
    easting_in: float,
    northing_in: float,
    northing_ref: float = 10000000.0,
) -> tuple[float, float]:
    """Inverse of :func:`utm_to_mgrs` for a single 100 km square reference.

    The MGRS row letter alone is ambiguous: the 20-letter alphabet repeats every
    2 000 km of northing, and the same letters are reused north and south of the
    equator.  ``northing_ref`` disambiguates by selecting the 2 000 km band and
    the hemisphere whose UTM northing is closest to it; it defaults to
    10 000 000 m (the northern hemisphere).
    """
    zone = int(zone)
    if not 1 <= zone <= 60:
        raise ValueError(f"invalid UTM zone {zone!r}")
    col = _MGRS_COL.tolist().index(col_letter.upper())
    col_band = (col - 8 * ((zone - 1) % 3)) % 24
    easting = 100000.0 * (col_band + 1) + float(easting_in)

    row = _MGRS_ROW.tolist().index(row_letter.upper())
    band0 = (row - 5 * (1 - zone % 2)) % 20
    best, best_err = None, np.inf
    for k in range(-8, 9):  # candidate 2 000 km bands, north and south of the equator
        band = band0 + 20 * k
        for offset in (0.0, 10000000.0):  # southern / northern false northing
            northing = band * 100000.0 + float(northing_in) + offset
            err = abs(northing - float(northing_ref))
            if err < best_err:
                best, best_err = northing, err
    return easting, best
