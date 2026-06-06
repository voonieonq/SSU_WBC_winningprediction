"""날씨 → 예상 득점 보정 (Open-Meteo seasonal forecast)."""

from __future__ import annotations

from typing import Optional, Tuple

import requests


def fetch_temperature_run_multiplier(
    lat: float,
    lon: float,
    timeout: float = 8.0,
) -> Tuple[float, str]:
    """
    기온 기반 득점 배율 (명세: 날씨 → 예상득점 → 승패).
    - 이상적 야구 온도 ~22°C 근처
    - 추위/폭염은 득점 소폭 감소 가정
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        temps = data.get("hourly", {}).get("temperature_2m", [])
        if not temps:
            return 1.0, "기온 데이터 없음 — 중립(1.0) 적용"
        t_avg = sum(temps[:24]) / min(24, len(temps))
        # 22°C 최적, 편차 10°C당 약 4% 득점 변화
        delta = abs(t_avg - 22.0)
        mult = max(0.88, min(1.08, 1.0 - delta * 0.004))
        wind_note = ""
        return mult, f"평균 기온 {t_avg:.1f}°C → 득점배율 {mult:.3f}{wind_note}"
    except Exception as e:
        return 1.0, f"Open-Meteo 조회 실패({e}) — 중립(1.0)"


def manual_weather_multiplier(temp_c: float, wind_kmh: float = 0.0) -> Tuple[float, str]:
    """API 없을 때 수동 입력."""
    delta = abs(temp_c - 22.0)
    mult = max(0.88, min(1.08, 1.0 - delta * 0.004))
    # 강풍은 장타 억제
    if wind_kmh > 25:
        mult *= 0.97
    return mult, f"수동: {temp_c}°C, 바람 {wind_kmh}km/h → {mult:.3f}"
